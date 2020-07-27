import os
from os import path, mkdir, listdir, makedirs
import sys
import shutil
import random
import math
import gdal
import glob
import timeit
import copy
import argparse 
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, Queue, Process
from functools import partial
from math import ceil

import rasterio
from rasterio import features
from affine import Affine

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

import skimage
from skimage import measure, io
from skimage.morphology import square, erosion, dilation, remove_small_objects, watershed, remove_small_holes
from skimage.color import label2rgb
from scipy import ndimage
from shapely.wkt import dumps, loads
from shapely.geometry import shape, Polygon
from imgaug import augmenters as iaa
import cv2

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from geffnet.efficientnet_builder import *
#import selim_sef_sn4
import base

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
#cudnn.enabled = config.CUDNN.ENABLED

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


################################# MODEL 

class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(probs, dim=2)# batch x k x hw
        return torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3).contiguous() # batch x k x c x 1

class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, inc, keyc, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__()
        self.keyc = keyc
        self.f_pixel = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU(), nn.Conv2d(keyc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_object = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU(), nn.Conv2d(keyc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_down = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_up = nn.Sequential(nn.Conv2d(keyc, inc, 1, bias=False), nn.BatchNorm2d(inc), nn.ReLU())

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        query = self.f_pixel(x).view(batch_size, self.keyc, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.keyc, -1)
        value = self.f_down(proxy).view(batch_size, self.keyc, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.keyc**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.keyc, *x.size()[2:])
        context = self.f_up(context)
        return context

class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels)
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return self.conv_bn_dropout(torch.cat([context, feats], 1))

def m(in_channels, out_channels, k, d=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, k, padding=d if d>1 else k//2, dilation=d, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels = 256, rates = [12, 24, 36]):
        super(ASPP, self).__init__()

        self.c1 = m(in_channels, out_channels, 1)
        self.c2 = m(in_channels, out_channels, 3, rates[0])
        self.c3 = m(in_channels, out_channels, 3, rates[1])
        self.c4 = m(in_channels, out_channels, 3, rates[2])
        self.cg = nn.Sequential(nn.AdaptiveAvgPool2d(1), m(in_channels, out_channels, 1))
        self.project = m(4 * out_channels, out_channels, 1)
        self.projectg = m(out_channels, out_channels, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(x)
        cg = self.cg(x)
        c14 = self.project(torch.cat([c1, c2, c3, c4], dim=1))
        cg = self.projectg(cg)
        return self.drop(c14 + cg)


class GenEfficientNet(nn.Module):
    def __init__(self, block_args, num_classes=1000, in_chans=3, num_features=1280, stem_size=32, fix_stem=False, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_connect_rate=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog', dilations=[False, False,False,False]):
        super(GenEfficientNet, self).__init__()

        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        builder = EfficientNetBuilder(channel_multiplier, channel_divisor, channel_min, pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_connect_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args, dilations))

        self.conv_head = select_conv2d(builder.in_chs, num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for n, m in self.named_modules():
            if weight_init == 'goog':
                initialize_weight_goog(m, n)
            else:
                initialize_weight_default(m, n)


class Unet(nn.Module):
    def __init__(self, extra_num = 1, dec_ch = [32, 64, 128, 256, 1024], stride = 32, net='b5', bot1x1=False, glob=False, bn = False, aspp=False, ocr=False, aux = False):
        super().__init__()

        self.extra_num = extra_num
        self.stride = stride
        self.bot1x1 = bot1x1
        self.glob = glob
        self.bn = bn
        self.aspp = aspp
        self.ocr = ocr
        self.aux = aux

        if net == 'b4':
            channel_multiplier=1.4
            depth_multiplier=1.8
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth'
            enc_ch = [24, 32, 56, 160, 1792]
        if net == 'b5':
            channel_multiplier=1.6
            depth_multiplier=2.2
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth'
            enc_ch = [24, 40, 64, 176, 2048]
        if net == 'b6':
            channel_multiplier=1.8
            depth_multiplier=2.6
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth'
            enc_ch = [32, 40, 72, 200, 2304]
        if net == 'b7':
            channel_multiplier=2.0
            depth_multiplier=3.1
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth'
            enc_ch = [32, 48, 80, 224, 2560]
        if net == 'l2':
            channel_multiplier=4.3
            depth_multiplier=5.3
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth'
            enc_ch = [72, 104, 176, 480, 5504]

        dilations = [False, False, False, False]
        if stride == 16:
            dec_ch[4] = enc_ch[4]
            dilations = [False, False, False, True]
        elif stride == 8:
            dec_ch[3] = enc_ch[4]
            dilations = [False, False, True, True]

        def mod(cin, cout, k=3):
            if bn:
                return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k//2, bias=False), torch.nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            else:
                return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k//2), nn.ReLU(inplace=True))

        if self.aspp:
            self.asppc = ASPP(enc_ch[4], 256)
            enc_ch[4] = 256
        if self.ocr:
            midc = 512
            keyc = 256
            numcl = 4 * 4 * 3
            enc_ch[4] = 512
            dec_ch[2] = midc
            inpc = sum(enc_ch[1:])
            self.aux_head = nn.Sequential(nn.Conv2d(inpc, inpc, 3, padding=1, bias=False), nn.BatchNorm2d(inpc), nn.ReLU(inplace=True), nn.Conv2d(inpc, numcl, 1))
            self.conv3x3_ocr = nn.Sequential(nn.Conv2d(inpc, midc, 3, padding=1, bias=False), nn.BatchNorm2d(midc), nn.ReLU(inplace=True))
            self.ocr_gather_head = SpatialGather_Module(numcl)
            self.ocr_distri_head = SpatialOCR_Module(in_channels=midc, key_channels=keyc, out_channels=midc, dropout=0.05)
        if self.glob:
            self.global_f = nn.Sequential(nn.AdaptiveAvgPool2d(1), mod(enc_ch[4], dec_ch[4], 1))

        self.bot0extra = mod(206, enc_ch[4])
        self.bot1extra = mod(206, dec_ch[4])
        self.bot2extra = mod(206, dec_ch[3])
        self.bot3extra = mod(206, dec_ch[2])
        self.bot4extra = mod(206, dec_ch[1])
        self.bot5extra = mod(206, 6)

        self.dec0 = mod(enc_ch[4], dec_ch[4])
        self.dec1 = mod(dec_ch[4], dec_ch[3])
        self.dec2 = mod(dec_ch[3], dec_ch[2])
        self.dec3 = mod(dec_ch[2], dec_ch[1])
        self.dec4 = mod(dec_ch[1], dec_ch[0])

        if self.bot1x1:
            self.bot1x10 = mod(enc_ch[3], enc_ch[3], 1)
            self.bot1x11 = mod(enc_ch[2], enc_ch[2], 1)
            self.bot1x12 = mod(enc_ch[1], enc_ch[1], 1)
            self.bot1x13 = mod(enc_ch[0], enc_ch[0], 1)

        self.bot0 = mod(enc_ch[3] + dec_ch[4], dec_ch[4])
        self.bot1 = mod(enc_ch[2] + dec_ch[3], dec_ch[3])
        self.bot2 = mod(enc_ch[1] + dec_ch[2], dec_ch[2])
        self.bot3 = mod(enc_ch[0] + dec_ch[1], dec_ch[1])

        self.up = nn.Upsample(scale_factor=2)
        self.upps = nn.PixelShuffle(upscale_factor=2)
        self.final = nn.Conv2d(dec_ch[0], 6, 1)
        if self.aux:
            aux_c = max(enc_ch[3], 16 * 16 * 3)
            self.aux_final = nn.Sequential(nn.Conv2d(enc_ch[3], aux_c, 3, padding=1, bias=False), nn.BatchNorm2d(aux_c), nn.ReLU(inplace=True), nn.Conv2d(aux_c, 16 * 16 * 3, 1))

        self._initialize_weights()

        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25']]
        enc = GenEfficientNet(in_chans=3, block_args=decode_arch_def(arch_def, depth_multiplier), num_features=round_channels(1280, channel_multiplier, 8, None), stem_size=32,
            channel_multiplier=channel_multiplier, act_layer=resolve_act_layer({}, 'swish'), norm_kwargs=resolve_bn_args({'bn_eps': BN_EPS_TF_DEFAULT}), pad_type='same', dilations=dilations)
        state_dict = load_state_dict_from_url(url)
        enc.load_state_dict(state_dict, strict=True)

        stem_size = round_channels(32, channel_multiplier, 8, None)
        conv_stem = select_conv2d(4, stem_size, 3, stride=2, padding='same')
        _w = enc.conv_stem.state_dict()
        _w['weight'] = torch.cat([_w['weight'], _w['weight'][:,1:2] ], 1)
        conv_stem.load_state_dict(_w)

        self.enc0 = nn.Sequential(conv_stem, enc.bn1, enc.act1, enc.blocks[0])
        self.enc1 = nn.Sequential(enc.blocks[1])
        self.enc2 = nn.Sequential(enc.blocks[2])
        self.enc3 = nn.Sequential(enc.blocks[3], enc.blocks[4])
        self.enc4 = nn.Sequential(enc.blocks[5], enc.blocks[6], enc.conv_head, enc.bn2, enc.act2)
        if self.ocr:
            self.enc4 = nn.Sequential(enc.blocks[5], enc.blocks[6])


    def forward(self, x, strip, direction, coord):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        if self.bot1x1:
            enc3 = self.bot1x10(enc3)
            enc2 = self.bot1x11(enc2)
            enc1 = self.bot1x12(enc1)
            enc0 = self.bot1x13(enc0)

        ex = torch.cat([strip, direction, coord], 1)
        x = enc4
        if self.aspp:
            x = self.asppc(x)
        elif self.ocr:
            enc1 = enc1
            enc2 = self.up(enc2)
            enc3 = self.up(self.up(enc3))
            enc4 = self.up(self.up(self.up(enc4)))
            feats = torch.cat([enc4, enc3, enc2, enc1], 1)

            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)
            cont = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, cont)
  
            x = self.dec3(feats)
            x = torch.cat([x, enc0], dim=1)
            x = self.bot3(x)
            x = self.dec4(x)
            return self.final(x), self.upps(self.upps(out_aux))

        if self.stride == 32:
            x = self.dec0(self.up(x + (0 if self.extra_num <= 0 else self.bot0extra(ex)))) + (self.global_f(x) if self.glob else 0)
            x = torch.cat([x, enc3], dim=1)
            x = self.bot0(x) 
        if self.stride == 32 or self.stride == 16:
            x = self.dec1(self.up(x + (0 if self.extra_num <= 1 else self.bot1extra(ex))))
            x = torch.cat([x, enc2], dim=1)
            x = self.bot1(x)
        x = self.dec2(self.up(x + (0 if self.extra_num <= 2 else self.bot2extra(ex))))
        x = torch.cat([x, enc1], dim=1)
        x = self.bot2(x)
        x = self.dec3(self.up(x + (0 if self.extra_num <= 3 else self.bot3extra(ex))))
        x = torch.cat([x, enc0], dim=1)
        x = self.bot3(x) 
        x = self.dec4(self.up(x + (0 if self.extra_num <= 4 else self.bot4extra(ex))))
        x = self.final(x) + (0 if self.extra_num <= 5 else self.bot5extra(ex))
        if self.aux:
            return x, self.upps(self.upps(self.upps(self.upps(self.aux_final(enc3)))))
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

############################## DATASET

def _blend(img1, img2, alpha):
    return img1 * alpha + (1 - alpha) * img2

_alpha = np.asarray([0.25, 0.25, 0.25, 0.25]).reshape((1, 1, 4))
def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)

def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)

def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)

def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 4)
    return _blend(img, gs, alpha)

def parse_img_id(file_path, orients):
    file_name = file_path.split('/')[-1]
    stripname = '_'.join(file_name.split('_')[-4:-2])

    direction = int(orients.loc[stripname]['direction'])
    direction = torch.from_numpy(np.reshape(np.asarray([direction]), (1,1,1))).float()

    val = int(orients.loc[stripname]['val'])
    strip = torch.Tensor(np.zeros((len(orients.index), 1, 1))).float()
    strip[val] = 1

    coord = np.asarray([orients.loc[stripname]['coord_y']])
    coord = torch.from_numpy(np.reshape(coord, (1,1,1))).float() - 0.5
    return direction, strip, coord


class MyData(Dataset):
    def __init__(self, image_paths, label_paths, train, color = False, crop_size = None, reorder_bands = 0,
        rot_prob = 0.3, scale_prob = 0.5, color_aug_prob = 0.0, gauss_aug_prob = 0.0, flipud_prob=0.0, fliplr_prob = 0.0, rot90_prob = 0.0, gamma_aug_prob = 0.0, elastic_aug_prob = 0.0, 
            channel_swap_prob = 0.0, train_min_building_size=0):
        super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.train = train
        self.color = color
        self.crop_size = crop_size
        self.reorder_bands = reorder_bands
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.color_aug_prob = color_aug_prob
        self.gamma_aug_prob = gamma_aug_prob
        self.gauss_aug_prob = gauss_aug_prob
        self.elastic_aug_prob = elastic_aug_prob
        self.flipud_prob = flipud_prob
        self.fliplr_prob = fliplr_prob
        self.rot90_prob = rot90_prob
        self.channel_swap_prob = channel_swap_prob
        self.train_min_building_size = train_min_building_size
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.orients = pd.read_csv(rot_out_path, index_col = 0)
        self.orients['val'] = list(range(len(self.orients.index)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.color:
            img = skimage.io.imread(os.path.join('/data/train/AOI_11_Rotterdam/PS-RGB', os.path.basename(self.image_paths[idx]).replace('SAR-Intensity', 'PS-RGB') )) 
            img = img[:, :, [2,0,0,1]] 
        else:
            img = skimage.io.imread(self.image_paths[idx]) #, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_COLOR
        m = np.where((img.sum(axis=2) > 0).any(1))
        ymin, ymax = np.amin(m), np.amax(m) + 1
        m = np.where((img.sum(axis=2) > 0).any(0))
        xmin, xmax = np.amin(m), np.amax(m) + 1
        img = img[ymin:ymax, xmin:xmax]

        if self.train:
            msk = skimage.io.imread(self.label_paths[idx]) #, cv2.IMREAD_UNCHANGED)
            #pan = skimage.io.imread(os.path.join('/data/train/AOI_11_Rotterdam/PAN', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PAN') )) 
            #rgb = skimage.io.imread(os.path.join('/data/train/AOI_11_Rotterdam/PS-RGB', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGB') )) 
            #rgb = skimage.io.imread(os.path.join('/data/train/AOI_11_Rotterdam/PS-RGBNIR', os.path.basename(self.label_paths[idx]).replace('SAR-Intensity', 'PS-RGBNIR') )) 
            #rgb = np.concatenate([rgb, pan], axis=2)
            msk = msk[ymin:ymax, xmin:xmax]
            #rgb = rgb[ymin:ymax, xmin:xmax]

            pad = max(0, self.crop_size - img.shape[0])
            img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            msk = cv2.copyMakeBorder(msk, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)
            #rgb = cv2.copyMakeBorder(rgb, 0, pad, 0, 0, cv2.BORDER_CONSTANT, 0.0)

            if random.random() < args.rot_prob:
                rot_mat = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), random.randint(0, 10) - 5, 1.0)
                img = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                #rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            if random.random() < args.scale_prob:
                rot_mat = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), 0, random.uniform(0.5,2.0))
                img = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                msk = cv2.warpAffine(msk, rot_mat, msk.shape[:2], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                #rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            x0 = random.randint(0, img.shape[1] - self.crop_size)
            y0 = random.randint(0, img.shape[0] - self.crop_size)
            img = img[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            msk = msk[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            #rgb = rgb[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

            if random.random() < self.color_aug_prob:
                img = saturation(img, 0.8 + random.random() * 0.4)
            if random.random() < self.color_aug_prob:
                img = brightness(img, 0.8 + random.random() * 0.4)
            if random.random() < self.color_aug_prob:
                img = contrast(img, 0.8 + random.random() * 0.4)
            if random.random() < self.gamma_aug_prob:
                gamma = 0.8  + 0.4 * random.random()
                img = np.clip(img, a_min = 0.0, a_max = None)
                img = np.power(img, gamma)
            if random.random() < args.gauss_aug_prob:
                gauss = np.random.normal(10.0, 10.0**0.5 , img.shape)
                img += gauss - np.min(gauss)
            if random.random() < args.elastic_aug_prob:
                el_det = self.elastic.to_deterministic()
                img = el_det.augment_image(img)
            if random.random() < self.flipud_prob:
                img = np.flipud(img)
                msk = np.flipud(msk)
                #rgb = np.flipud(rgb)
            if random.random() < self.fliplr_prob:
                img = np.fliplr(img)
                msk = np.fliplr(msk)
                #rgb = np.fliplr(rgb)
            if random.random() < self.rot90_prob:
                k = random.randint(0,3)
                img = np.rot90(img, k)
                msk = np.rot90(msk, k)
                #rgb = np.rot90(rgb, k)
            if random.random() < self.channel_swap_prob:
                c1 = random.randint(0,3)
                c2 = random.randint(0,3)
                img[:, :, [c1, c2]] = img[:, :, [c2, c1]]

        direction, strip, coord = parse_img_id(self.image_paths[idx], self.orients)
        if direction.item():
            img = np.fliplr(np.flipud(img))
            if self.train:
                msk = np.fliplr(np.flipud(msk))
                #rgb = np.fliplr(np.flipud(rgb))

        if self.color:
            img = (img - np.array([93.41131901, 97.27417209, 97.27417209, 102.25152583])) / np.array([38.8338671, 41.6705231, 41.6705231, 37.34136047])
        else:
            img = (img - np.array([28.62501827, 36.09922463, 33.84483687, 26.21196667])) / np.array([8.41487376, 8.26645475, 8.32328472, 8.63668993])

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        if self.reorder_bands == 1:
            img = img[[2,3,0,1]]
        elif self.reorder_bands == 2:
            img = img[[1,3,0,2]]
        elif self.reorder_bands == 3:
            img = img[[0,3,1,2]]

        if self.train:
            weights = np.ones_like(msk[:,:,:1], dtype=float)
            regionlabels, regioncount = measure.label(msk[:,:,0], background=0, connectivity=1, return_num=True)
            regionproperties = measure.regionprops(regionlabels)
            for bl in range(regioncount):
                if regionproperties[bl].area < self.train_min_building_size:
                    msk[:,:,0][regionlabels == bl+1] = 0
                    msk[:,:,1][regionlabels == bl+1] = 0
                weights[regionlabels == bl+1] = 1024.0 / regionproperties[bl].area

            msk[:, :, :3] = (msk[:, :, :3] > 1) * 1
            #msk[:, :, 3] -= 10.0
            #rgb = rgb / 255.0 - 0.5
            weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
            msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).float()
            #rgb = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
            rgb = torch.Tensor([0])
        else:
            msk = rgb = weights = regioncount = torch.Tensor([0])
        return {"img": img, "mask": msk, 'rgb': rgb, 'strip': strip, 'direction': direction, 'coord': coord, 'img_name': self.image_paths[idx],
                'ymin': ymin, 'xmin': xmin, 'b_count': regioncount, 'weights': weights}

#################################### EVAL

def test_postprocess(pred_folder, pred_csv, **kwargs):
    np.seterr(over = 'ignore')
    sourcefiles = sorted(glob.glob(os.path.join(pred_folder, '*')))
    with Pool() as pool:
        proposals = [p for p in tqdm(pool.imap_unordered(partial(test_postprocess_single, **kwargs), sourcefiles), total = len(sourcefiles))]
    pd.concat(proposals).to_csv(pred_csv, index=False)

def test_postprocess_single(sourcefile, watershed_line=True, conn = 2, polygon_buffer = 0.5, tolerance = 0.5, seed_msk_th = 0.75, area_th_for_seed = 110, pred_th = 0.5, area_th = 80,
        contact_weight = 1.0, edge_weight = 0.0, seed_contact_weight = 1.0, seed_edge_weight = 1.0):
    mask = gdal.Open(sourcefile).ReadAsArray() # logits
    mask = 1.0 / (1 + np.exp(-mask))
    mask[0] = mask[0] * (1 - contact_weight * mask[2]) * (1 - edge_weight * mask[1])

    seed_msk = mask[0] * (1 - seed_contact_weight * mask[2]) * (1 - seed_edge_weight * mask[1])
    seed_msk = measure.label((seed_msk > seed_msk_th), connectivity=conn, background=0)
    props = measure.regionprops(seed_msk)
    for i in range(len(props)):
        if props[i].area < area_th_for_seed:
            seed_msk[seed_msk == i + 1] = 0
    seed_msk = measure.label(seed_msk, connectivity=conn, background=0)

    mask = watershed(-mask[0], seed_msk, mask=(mask[0] > pred_th), watershed_line=watershed_line)
    mask = measure.label(mask, connectivity=conn, background=0).astype('uint8')

    polygon_generator = features.shapes(mask, mask)
    polygons = []
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(polygon_buffer)
        if p.area >= area_th:
            p = dumps(p.simplify(tolerance=tolerance), rounding_precision=0)
            #if "), (" in p:
            #    p = p.split('), (')[0] +  '))",' + p.split('))",')[-1]
            polygons.append(p)

    tilename = '_'.join(os.path.splitext(os.path.basename(sourcefile))[0].split('_')[-4:])
    csvaddition = pd.DataFrame({'ImageId': tilename, 'BuildingId': range(len(polygons)), 'PolygonWKT_Pix': polygons, 'Confidence': 1 })
    return csvaddition

def evaluation(pred_csv, gt_csv):
    evaluator = base.Evaluator(gt_csv)
    evaluator.load_proposal(pred_csv, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=80)
    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2*tp) / (2*tp + fp + fn)
    print('Validation F1 {} tp {} fp {} fn {}'.format(f1score, tp, fp, fn))
    return f1score

############################## PREPROCESS 

def polygon_to_mask(poly, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def process_image(img_path, segm_dir, edge_width, contact_width, gt_buildings_csv):
    gt_buildings = pd.read_csv(gt_buildings_csv)
    img_name = os.path.basename(img_path)
    vals = gt_buildings[( gt_buildings['ImageId'] == '_'.join(img_name.split('_')[-4:])[:-4] )][['TileBuildingId', 'PolygonWKT_Pix', 'Mean_Building_Height']].values
    labels = np.zeros((900, 900), dtype='uint16')
    heights = np.zeros((900, 900), dtype='float')
    cur_lbl = 0
    for i in range(vals.shape[0]):
        poly = loads(vals[i, 1])
        if not poly.is_empty:
            cur_lbl += 1
            msk = polygon_to_mask(poly, (900, 900))
            labels[msk > 0] = cur_lbl
            if vals[i,2] == vals[i,2]:
                heights[msk > 0] = vals[i,2]
    #skimage.io.imsave(path.join(segm_dir, img_name), labels)

    msk = np.zeros((900, 900, 3), dtype='uint8')
    if cur_lbl > 0:
        footprint_msk = labels > 0

        heights = np.clip(heights, a_min = 0.0, a_max = 255.0).astype('uint8')

        border_msk = np.zeros_like(labels, dtype='bool')
        for l in range(1, labels.max() + 1):
            tmp_lbl = labels == l
            _k = square(edge_width)
            tmp = erosion(tmp_lbl, _k)
            tmp = tmp ^ tmp_lbl
            border_msk = border_msk | tmp

        tmp = dilation(labels > 0, square(contact_width))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = tmp | border_msk
        tmp = dilation(tmp, square(contact_width))
        contact_msk = np.zeros_like(labels, dtype='bool')
        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    sz = 3
                else:
                    sz = 1
                uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
                if len(uniq[uniq > 0]) > 1:
                    contact_msk[y0, x0] = True

        msk = np.stack((255*footprint_msk, 255*border_msk, 255*contact_msk)).astype('uint8')
        #msk = np.stack((255*footprint_msk, 255*border_msk, 255*contact_msk, heights)).astype('uint8')
        msk = np.rollaxis(msk, 0, 3)

    skimage.io.imsave(path.join(segm_dir, img_name), msk)

############################### TRAIN 

class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets, weights = 1.0):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        weights = weights.contiguous()

        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        weights = weights.contiguous().view(-1)[non_ignored]

        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)

        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return ((-(1. - pt) ** self.gamma * torch.log(pt)) * weights).mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, eps = 1e-6):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        batch_size = outputs.size()[0]
        if not self.per_image:
            batch_size = 1
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + self.eps
        loss = (1 - (2 * intersection + self.eps) / union).mean()
        return loss

def load_state_dict(model, state_dict):
    missing_keys = [] 
    unexpected_keys = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, [])
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(model)
    print('Unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    print('Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))


rot_in_path = '/root/SAR_orientations.txt'
rot_out_path = '/root/SAR_orientations.csv'
models_folder = '/wdata/weights'
merged_pred_folder = '/wdata/merged_pred'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
    parser.add_argument('--split_folds', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--val_search', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge_folds', nargs='+', default=[0, 1, 2, 3, 6, 7, 8, 9], type=int)

    parser.add_argument('--train_data_folder', default='/data/train/AOI_11_Rotterdam/', type=str)
    parser.add_argument('--test_data_folder', default='/data/test_public/AOI_11_Rotterdam/', type=str)
    parser.add_argument('--folds_path', default='/root/folds.csv', type=str)
    parser.add_argument('--gt_csv', default='/root/gt_fold_{0}_csv', type=str)
    parser.add_argument('--segm_dir', default='/wdata/masks_cannab', type=str)
    parser.add_argument('--pred_csv', default='/wdata/pred_fold_{0}_csv', type=str)
    parser.add_argument('--pred_folder', default='/wdata/pred_fold_{0}_0', type=str)
    parser.add_argument('--snapshot_last', default='snapshot_fold_{0}_last', type=str)
    parser.add_argument('--snapshot_best', default='snapshot_fold_{0}_best', type=str)
    parser.add_argument('--solution_file', default='/wdata/solution.csv', type=str)

    parser.add_argument('--edge_width', default=3, type=int)
    parser.add_argument('--contact_width', default=9, type=int)
    parser.add_argument('--train_min_building_size', default=0, type=int)

    parser.add_argument('--fold', default=9, type=int)
    parser.add_argument('--start_val_epoch', default=20, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--apex', action='store_true')

    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--warm_up_lr_scale', default=1.0, type=float)
    parser.add_argument('--warm_up_lr_epochs', default=0, type=int)
    parser.add_argument('--warm_up_dec_epochs', default=0, type=int)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--pos_weight', default=0.5, type=float)
    parser.add_argument('--b_count_weight', default=0.5, type=float)
    parser.add_argument('--b_count_div', default=8, type=float)
    parser.add_argument('--b_rev_size_weight', default=0.0, type=float)
    parser.add_argument('--focal_weight', default=1.0, type=float)
    parser.add_argument('--edge_weight', default=0.25, type=float)
    parser.add_argument('--contact_weight', default=0.1, type=float)
    parser.add_argument('--height_scale', default=0.0, type=float)
    parser.add_argument('--rgb_weight', default=0.0, type=float)
    parser.add_argument('--loss_eps', default=1e-6, type=float)
    parser.add_argument('--clip_grad_norm_value', default=1.2, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--rot_prob', default=0.7, type=float)
    parser.add_argument('--scale_prob', default=1.0, type=float)
    parser.add_argument('--color_aug_prob', default=1.0, type=float)
    parser.add_argument('--gauss_aug_prob', default=0.0, type=float)
    parser.add_argument('--gamma_aug_prob', default=0.0, type=float)
    parser.add_argument('--elastic_aug_prob', default=0.0, type=float)
    parser.add_argument('--flipud_prob', default=0.0, type=float)
    parser.add_argument('--fliplr_prob', default=0.5, type=float)
    parser.add_argument('--rot90_prob', default=0.0, type=float)
    parser.add_argument('--channel_swap_prob', default=0.0, type=float)
    parser.add_argument('--input_scale', default=1.0, type=float)
    parser.add_argument('--strip_scale', default=1.0, type=float)
    parser.add_argument('--direction_scale', default=1.0, type=float)
    parser.add_argument('--coord_scale', default=1.0, type=float)
    parser.add_argument('--reorder_bands', default=3, type=int)
    parser.add_argument('--extra_num', default=1, type=int)
    parser.add_argument('--stride', default=32, type=int)
    parser.add_argument('--bot1x1', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--glob', action='store_true')
    parser.add_argument('--aspp', action='store_true')
    parser.add_argument('--ocr', action='store_true')
    parser.add_argument('--net', default='b5')
    parser.add_argument('--dec_ch', nargs='+', default=[32, 64, 128, 256, 1024], type=int)
    parser.add_argument('--aux_scale', default=0.0, type=float)

    args = parser.parse_args(sys.argv[1:])
    if '{0}' not in args.gt_csv:
        args.gt_csv = args.gt_csv + 'fold_{0}.csv'

    if args.split_folds:
        gt_buildings_csv = os.path.join(args.train_data_folder, 'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
        gt_buildings = pd.read_csv(gt_buildings_csv)
        makedirs(args.segm_dir, exist_ok=True)
        sar_paths = glob.glob(os.path.join(args.train_data_folder, 'SAR-Intensity', '*.tif'))
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(partial(process_image, segm_dir=args.segm_dir, edge_width=args.edge_width, contact_width=args.contact_width, gt_buildings_csv=gt_buildings_csv), sar_paths)):
                pass
        sys.exit(0)

        rotations = pd.read_csv(rot_in_path, sep=' ', index_col = 0, names=['strip', 'direction'], header=None)
        rotations['sum_y'] = 0.0
        rotations['ctr_y'] = 0.0

        df_fold = pd.DataFrame(columns=['ImageId', 'sar', 'segm', 'rotation', 'x', 'y', 'fold'])
        ledge = 591640
        redge = 596160
        numgroups = 10
        for sar_path in tqdm(sar_paths):
            ImageId = '_'.join(os.path.splitext(os.path.basename(sar_path))[0].split('_')[-4:])
            stripname = '_'.join(ImageId.split('_')[-4:-2])
            rotation = rotations.loc[stripname]['direction'].squeeze()
            tr = gdal.Open(sar_path).GetGeoTransform()
            rotations.loc[stripname, 'sum_y'] += tr[3]
            rotations.loc[stripname, 'ctr_y'] += 1
            fold_no = min(numgroups-1, max(0, math.floor((tr[0]-ledge) / (redge-ledge) * numgroups)))
            segm_path = os.path.join(args.segm_dir, os.path.basename(sar_path)) #.replace('tif', 'png'))
            df_fold = df_fold.append({'ImageId': ImageId, 'sar': sar_path, 'segm': segm_path, 'rotation': rotation, 'x': tr[0], 'y': tr[3], 'fold': fold_no}, ignore_index=True)
        df_fold.to_csv(args.folds_path, index=False)

        for i in range(numgroups):
            print( '%i: %i' % (i, len(df_fold[df_fold['fold']==i])))
            img_ids = df_fold[df_fold['fold'] == i]['ImageId'].values
            gt_buildings[gt_buildings.ImageId.isin(img_ids)].to_csv(args.gt_csv.format(i), index=False)

        rotations['mean_y'] = rotations['sum_y'] / rotations['ctr_y']
        rotations['coord_y'] = (((rotations['mean_y'] - 5746153.106161971) / 11000) + 0.2)
        rotations.to_csv(rot_out_path, index=True)
        if not args.train:
            sys.exit(0)
    elif args.merge:
        shutil.rmtree(merged_pred_folder, ignore_errors=True)
        makedirs(merged_pred_folder, exist_ok=True)
        pred_folders = [args.pred_folder.format(i) for i in args.merge_folds]
        for filename in tqdm(listdir(pred_folders[0])):
            used_msks = [skimage.io.imread(path.join(ff, filename)) for ff in pred_folders]
            msk = np.zeros_like(used_msks[0], dtype='float')
            for used_msk in used_msks:
                msk += used_msk.astype('float') / len(args.merge_folds)
            #cv2.imwrite(path.join(merged_pred_folder, fid), msk.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            #skimage.io.imsave(path.join(merged_pred_folder, fid), msk.astype('uint8'))
            skimage.io.imsave(path.join(merged_pred_folder, filename), msk)
        test_postprocess(merged_pred_folder, args.solution_file)
    #elif args.merge_oof:
    #    merged_oof_folder = '/wdata/merged_oof'
    #    makedirs(merged_oof_folder, exist_ok=True)
    #    val_files = [args.pred_oof_folder + '/' + v for v in listdir(args.pred_oof_folder) if '.png' in v] #[os.path.basename(f) for f in df[df['fold'] < num_folds]['sar'].values]
    #    def process_image(f):
    #        msk = skimage.io.imread(f).astype('float')
    #        #cv2.imwrite(path.join(merged_oof_folder, os.path.basename(f)), msk.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #        skimage.io.imsave(path.join(merged_oof_folder, os.path.basename(f)), msk.astype('uint8'))
    #    with Pool() as pool:
    #        pool.map(process_image, val_files)
    if not (args.train or args.val or args.test):
        sys.exit(0)


    ############# TRAINING

    args.gt_csv = args.gt_csv.format(args.fold%10)
    args.pred_csv = args.pred_csv.format(args.fold)
    args.pred_folder = args.pred_folder.format(args.fold)
    args.snapshot_last = args.snapshot_last.format(args.fold)
    args.snapshot_best = args.snapshot_best.format(args.fold)

    df = pd.read_csv(args.folds_path)
    train_img_files = [os.path.join(args.train_data_folder, 'SAR-Intensity', os.path.basename(x)) for x in df[np.logical_or(df['fold'] > (args.fold%10) + 1, df['fold'] < (args.fold%10) - 1)]['sar'].values]
    train_label_files = [os.path.join(args.segm_dir, os.path.basename(x)) for x in df[np.logical_or(df['fold'] > (args.fold%10) + 1, df['fold'] < (args.fold%10) - 1)]['segm'].values]
    if args.val:
        val_img_files = [os.path.join(args.train_data_folder, 'SAR-Intensity', os.path.basename(x)) for x in df[df['fold'] == (args.fold%10)]['sar'].values]
    elif args.test:
        val_img_files = glob.glob(os.path.join(args.test_data_folder, 'SAR-Intensity', '*.tif'))

    makedirs(models_folder, exist_ok=True)

    data_train = MyData(train_img_files, train_label_files, train=True, color=args.color, crop_size=args.crop_size, reorder_bands = args.reorder_bands,
            rot_prob = args.rot_prob, scale_prob = args.scale_prob, color_aug_prob = args.color_aug_prob, gauss_aug_prob=args.gauss_aug_prob, channel_swap_prob = args.channel_swap_prob,
            flipud_prob = args.flipud_prob, fliplr_prob = args.fliplr_prob, rot90_prob = args.rot90_prob, gamma_aug_prob=args.gamma_aug_prob, elastic_aug_prob = args.elastic_aug_prob,
            train_min_building_size = args.train_min_building_size)
    train_data_loader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True) #, worker_init_fn=np.random.seed(0))
    val_data_loader = DataLoader(MyData(val_img_files, None, train=False, color=args.color, reorder_bands = args.reorder_bands), batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    model = Unet(extra_num = args.extra_num, dec_ch = args.dec_ch, stride=args.stride, net=args.net, bot1x1 = args.bot1x1, glob=args.glob, bn=args.bn, aspp=args.aspp,
        ocr=args.ocr, aux = args.aux_scale > 0).cuda()
    if not args.train:
        loaded = torch.load(path.join(models_folder, args.snapshot_best))
        print("loaded checkpoint '{}' (epoch {}, f1 score {})".format(args.snapshot_best, loaded['epoch'], loaded['best_score']))
        load_state_dict(model, loaded['state_dict'])

    if args.val_search:
        for pbuffer in [0.0]: #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1.0]: #[0.0]:
            for tolerance in [0.5]: #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #[0.5]
                for seed_msk_th in [0.7, 0.75, 0.8]: #[0.75]: # top 0.8 albo 0.75
                    for area_th_for_seed in [80]: #[80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300]: #[80] # top 160, check 170
                        for pred_th in [0.5]:
                            for area_th in [80]: #[80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]: #[80]:
                                for seed_contact_weight in [1.0]: # [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]: #[1.0]:
                                    for seed_edge_weight in [1.0]: #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: #[1.0]:
                                        for contact_weight in [0.0, 1.0]: #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #[0.0]:
                                            for edge_weight in [0.0]: #[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: #[0.0]:
                                                test_postprocess(args.pred_folder, args.pred_csv, polygon_buffer=pbuffer, tolerance=tolerance, seed_msk_th=seed_msk_th, area_th_for_seed=area_th_for_seed,
                                                    pred_th=pred_th, area_th=area_th, contact_weight=contact_weight, edge_weight=edge_weight,
                                                    seed_contact_weight=seed_contact_weight, seed_edge_weight=seed_edge_weight)
                                                evaluation(args.pred_csv, args.gt_csv)
                                                print('pbuffer', pbuffer, 'tolerance', tolerance, 'seed mask th', seed_msk_th, 'area_th_for_seed', area_th_for_seed, 'pred_th', pred_th,
                                                    'area_th', area_th, 'contact weight', contact_weight, 'edge weight', edge_weight,
                                                    'seed_contact_weight', seed_contact_weight, 'seed_edge_weight', seed_edge_weight)

        sys.exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.apex:
        from apex import amp
        from apex.optimizers import FusedAdam
        #optimizer = FusedAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    def lr_comp(epoch):
        if epoch < args.warm_up_lr_epochs:
            return args.warm_up_lr_scale
        elif epoch < 60:
            return 1.0
        elif epoch < 80:
            return 0.33
        elif epoch < 90:
            return 0.1
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_comp)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100, 120], gamma=args.gamma)
    dice_loss = DiceLoss(eps=args.loss_eps).cuda()
    focal_loss = FocalLoss2d(gamma = args.focal_gamma, eps=args.loss_eps).cuda()
    l1_loss = torch.nn.SmoothL1Loss().cuda()
 
    q = Queue()
    best_f1 = -1.0

    for epoch in range(140 if args.train else 1):
        if args.train:
            if epoch < args.warm_up_dec_epochs:
                for m in [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4]:
                    for p in m.parameters(): 
                        p.requires_grad = False
            else:
                for m in [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4]:
                    for p in m.parameters(): 
                        p.requires_grad = True
 
            time2 = time.time()
            iterator = tqdm(train_data_loader)
            model.train()
            torch.cuda.empty_cache()
            for sample in iterator:
                time1 = time.time()
                load = time1 - time2

                imgs = args.input_scale * sample["img"].cuda(non_blocking=True)
                #rgb = sample['rgb'].cuda(non_blocking=True)
                strip = args.strip_scale * sample["strip"].cuda(non_blocking=True)
                direction = args.direction_scale * sample["direction"].cuda(non_blocking=True)
                coord = args.coord_scale * sample["coord"].cuda(non_blocking=True)
                target = sample["mask"].cuda(non_blocking=True)
                b_count = sample["b_count"].cuda(non_blocking=True) / args.b_count_div
                b_weights = b_count * args.b_count_weight + 1.0 * (1.0 - args.b_count_weight)
                b_rev_size_weights = sample["weights"].cuda(non_blocking=True)
                b_rev_size_weights = b_rev_size_weights * args.b_rev_size_weight + 1.0 * (1.0 - args.b_rev_size_weight)

                weights = torch.ones(size=target.shape).cuda()
                weights[target > 0.0] *= args.pos_weight
                weights[:, :1] *= b_rev_size_weights
                weights[:, 1:2] *= b_rev_size_weights
                for i in range(weights.shape[0]):
                    weights[i] = weights[i] * b_weights[i]

                outputs = model(imgs, strip, direction, coord)
                if isinstance(outputs, tuple):
                    output = outputs[1]
                    l0 = args.focal_weight * focal_loss(output[:, 0], target[:, 0], weights[:, 0]) + dice_loss(output[:, 0], target[:, 0])
                    l1 = args.edge_weight * (args.focal_weight * focal_loss(output[:, 1], target[:, 1], weights[:, 1]) + dice_loss(output[:, 1], target[:, 1]))
                    l2 = args.contact_weight * (args.focal_weight * focal_loss(output[:, 2], target[:, 2], weights[:, 2]) + dice_loss(output[:, 2], target[:, 2]))
                    l_height = 0 #l1_loss(target[:, 0] * output[:, 3], target[:, 0] * target[:, 3] * args.height_scale)
                    lrgb = 0 #l1_loss(output[:, 3:(3 + rgb.shape[1])], rgb * args.rgb_weight)
                    loss_aux = l0 + l1 + l2 + l_height + lrgb
                    output = outputs[0]
                else:
                    loss_aux = 0
                    output = outputs

                l0 = args.focal_weight * focal_loss(output[:, 0], target[:, 0], weights[:, 0]) + dice_loss(output[:, 0], target[:, 0])
                l1 = args.edge_weight * (args.focal_weight * focal_loss(output[:, 1], target[:, 1], weights[:, 1]) + dice_loss(output[:, 1], target[:, 1]))
                l2 = args.contact_weight * (args.focal_weight * focal_loss(output[:, 2], target[:, 2], weights[:, 2]) + dice_loss(output[:, 2], target[:, 2]))
                l_height = 0 #l1_loss(target[:, 0] * output[:, 3], target[:, 0] * target[:, 3] * args.height_scale)
                lrgb = 0 #l1_loss(output[:, 3:(3 + rgb.shape[1])], rgb * args.rgb_weight)
                loss = l0 + l1 + l2 + l_height + lrgb + args.aux_scale * loss_aux

                optimizer.zero_grad()
                if args.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad_norm_value)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm_value)
                optimizer.step()

                time2 = time.time()
                proc = time2 - time1
                iterator.set_description("epoch: {}; lr {:.7f}; Loss {:.4f} l0 {:.4f} l1 {:.4f} l2 {:.4f} lrgb {:.4f} l_height {:.4f} load time {:.4f} proc time {:.4f}".format(
                    epoch, scheduler.get_lr()[-1], loss, l0, l1, l2, lrgb, l_height, load, proc))

            scheduler.step()
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, path.join(models_folder, args.snapshot_last))
            torch.cuda.empty_cache()

        if args.val and epoch > args.start_val_epoch:
            t.join()
            best_f1 = max(best_f1, q.get())

        if (args.val and epoch >= args.start_val_epoch) or args.test:
            print('Validation starts')
            shutil.rmtree(args.pred_folder, ignore_errors=True)
            makedirs(args.pred_folder, exist_ok=True)
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _, sample in enumerate(tqdm(val_data_loader)):
                    mask = sample["mask"].numpy()
                    imgs = args.input_scale * sample["img"].cuda(non_blocking=True)
                    ymin, xmin = sample['ymin'].item(), sample['xmin'].item()
                    strip = sample["strip"].cuda(non_blocking=True)
                    direction = sample["direction"].cuda(non_blocking=True)
                    coord = sample["coord"].cuda(non_blocking=True)
                    _, _, h, w = imgs.shape

                    scales = [0.8, 1.0, 1.5]
                    flips = [lambda x: x, lambda x: torch.flip(x,(3,))]
                    rots = [(lambda x: torch.rot90(x,i,(2,3))) for i in range(0,1)]
                    rots2 = [(lambda x: torch.rot90(x,4-i,(2,3))) for i in range(0,1)]
                    oos = torch.zeros((imgs.shape[0], 6, imgs.shape[2], imgs.shape[3])).cuda()
                    for sc in scales:
                        im = F.interpolate(imgs, size=(ceil(h*sc/32)*32, ceil(w*sc/32)*32), mode = 'bilinear', align_corners=True)
                        for fl in flips:
                            for i, rot in enumerate(rots):
                                o = model(rot(fl(im)), args.strip_scale * strip, args.direction_scale * direction, args.coord_scale * coord)
                                if isinstance(o, tuple):
                                    o = o[0]
                                oos += F.interpolate(fl(rots2[i](o)), size=(h,w), mode = 'bilinear', align_corners=True)
                    o = oos / (len(scales) * len(flips) * len(rots))

                    o = np.moveaxis(o.cpu().data.numpy(), 1, 3)
                    for i in range(len(o)):
                        img = o[i][:,:,:3]
                        if direction[i].item():
                            img = np.fliplr(np.flipud(img))
                        img = cv2.copyMakeBorder(img, ymin, 900 - h - ymin, xmin, 900 - w - xmin, cv2.BORDER_CONSTANT, 0.0)
                        skimage.io.imsave(os.path.join(args.pred_folder, os.path.split(sample['img_name'][i])[1]), img)
            torch.cuda.empty_cache()

        if args.val and epoch >= args.start_val_epoch:
            to_save = {k: copy.deepcopy(v.cpu()) for k, v in model.state_dict().items()}
            def new_thread():
                test_postprocess(args.pred_folder, args.pred_csv)
                val_f1 = evaluation(args.pred_csv, args.gt_csv)
                print()
                print('    Validation loss at epoch {}: {:.5f}, best {}'.format(epoch, val_f1, max(val_f1, best_f1)))
                print()
                if best_f1 < val_f1 and args.train:
                    torch.save({'epoch': epoch, 'state_dict': to_save, 'best_score': val_f1}, path.join(models_folder, args.snapshot_best))
                q.put(val_f1)
            t = Process(target = new_thread)
            t.start()

    if args.val:
        t.join()
