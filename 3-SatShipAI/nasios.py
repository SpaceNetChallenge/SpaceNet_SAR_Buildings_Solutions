import os
import cv2
import collections
import time
import tqdm
from tqdm import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu

import segmentation_models_pytorch as smp

import gc
from scipy.ndimage.morphology import binary_fill_holes as bfh


from shapely.wkt import loads as wkt_loads
import shapely.wkt
import rasterio
import shapely
from rasterio import features
import shapely.geometry
import shapely.affinity
import geopandas as gpd


experiments1 = [
    {
        'exp_id': 'effNet4_weightedbce_border',
        'sizes': [480, 768, 768],
        'mode' : ['train', 'train', 'eval'],
        'activation': 'sigmoid',
        'encoder': 'efficientnet-b4',
        'bs': [10, 4, 4],
        'init_lr': 0.0001,
        'epochs': [100, 100, 30],
        'snapshots': [10, 10, 10],
        'weights': [1, 1, 2.0]
    },
    {
        'exp_id': 'srxt50_32x4_weightedbce_border',
        'sizes': [480, 768, 768],
        'mode': ['train', 'train', 'eval'],
        'activation': 'sigmoid',
        'encoder': 'se_resnext50_32x4d',
        'bs': [12, 5, 5],
        'init_lr': 0.0001,
        'epochs': [100, 100, 30],
        'snapshots': [10, 10, 10],
        'weights': [1, 1, 2.0]
    },

    ]

experiments2=[
    {
        'exp_id': 'dn201_weightedbce_border',
        'sizes': [480, 736, 736],
        'mode': ['train', 'train', 'eval'],
        'activation': 'sigmoid',
        'encoder': 'densenet201',
        'bs': [10, 4, 4],
        'init_lr': 0.0001,
        'epochs': [100, 100, 30],
        'snapshots': [10, 10, 10],
        'weights': [1, 1, 2.0]
    },
    {
        'exp_id': 'rsnt34_weightedbce_border',
        'sizes': [480, 896, 896],
        'mode': ['train', 'train', 'eval'],
        'activation': 'sigmoid',
        'encoder': 'resnet34',
        'bs': [12, 2, 2],
        'init_lr': 0.0001,
        'epochs': [100, 100, 30],
        'snapshots': [10, 10, 10],
        'weights': [0, 1, 2.0]
    }
]

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def get_training_augmentation(size=480):
    train_transform = [
        #             albu.HorizontalFlip(p=0.5),
        #             albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=0.3),
        albu.CLAHE(p=0.3),
        albu.ShiftScaleRotate(scale_limit=0.75, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        #             albu.RandomResizedCrop( 320, 640, scale=(0.5, 1.0), p=0.75),
        #             albu.RandomResizedCrop( 350, 525, scale=(0.5, 1.0), p=0.75),
        #             albu.Resize(350, 525)
        albu.Resize(size, size)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(size=480):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(size, size)
        #         albu.Resize(350, 525)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def multimask2mask3d_v3(multimask):
    num_buildings = len(np.unique(multimask))
    if multimask.sum() > 0:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], num_buildings - 1))
        #         for i in range(1,num_buildings):
        counter = 0
        for i in [x for x in np.unique(multimask) if x not in [0]]:
            mask3d[..., counter][multimask == i] = 1
            counter += 1
    else:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], 1))
    return (mask3d.astype('uint8'))


def mask2buildings(mask):
    maskC = mask.copy()
    maskC_output = np.zeros_like(maskC)  # .astype('int32')
    contours, hierarchy = cv2.findContours(maskC, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        maskC_output += (cv2.drawContours(maskC, [cnt], -1, 1, cv2.FILLED) > 127.5).astype('uint8')
    uns = np.unique(maskC_output).copy()
    for ii in range(len(uns)):
        maskC_output[maskC_output == uns[ii]] = ii

    return maskC_output


def masks2masknum_v2(masks):
    outmask = np.zeros(masks.shape[1:])
    add = masks.shape[0]
    for m in range(len(masks)):
        outmask += masks[m] * (m + 1 + add)
    un_masks = np.unique(outmask)
    for mm in range(len(un_masks)):
        outmask[outmask == un_masks[mm]] = mm
    return outmask  # .astype('uint8')


def masks2masknum(masks):
    outmask = np.zeros(masks.shape[1:])
    for m in range(len(masks)):
        outmask += masks[m] * (m + 1)
    return outmask

def mask_to_polygon(mask):
    all_polygons = []
    lens=[]
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask == 1), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
#         print(value)
#         print(len(shape['coordinates'][0]))
        all_polygons.append(shapely.geometry.shape(shape))
        lens.append(len(shape['coordinates'][0]))
#     print(np.argmax(lens))
    all_polygons = shapely.geometry.Polygon(all_polygons[np.argmax(lens)])
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
#         if all_polygons.type == 'Polygon':
#             all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def _convert_coordinates_to_raster(coords, img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    W1 = 1.0 * width * width / (width + 1)
    H1 = 1.0 * height * height / (height + 1)
    xf = W1 / x_max
    yf = H1 / y_max
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int



def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.int8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
#     print(interior_list)
    cv2.fillPoly(img_mask, perim_list, class_value)
#     img_mask[np.array(list(proposalcsv.PolygonWKT_Pix.values[-1].exterior.coords)).astype(int)]=0
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

def _get_and_convert_contours(onepolygon, raster_img_size, xymax):
    perim_list = []
    interior_list = []
#     if onepolygon is None:
#         return None
#     for k in range(len(onepolygon)):
    poly = onepolygon
#     for ppp in poly.interiors:
#         print(ppp)
    perim = np.array(list(poly.exterior.coords))
    perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
    perim_list.append(perim_c)
    for pi in poly.interiors:
        interior = np.array(list(pi.coords))
        interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
        interior_list.append(interior_c)
    return perim_list, interior_list

def polygon2mask(polygon, width, height):
    xymax = (900,900)

    mask = np.zeros(( width, height))

#     for i, p in enumerate(polygons):
    i=0
    polygon_list = wkt_loads(str(polygon))
#     if polygon_list.length == 0:
#         continue
    contours = _get_and_convert_contours(polygon_list, (width, height), xymax)
    mask = _plot_mask_from_contours((width, height), contours, 1)
    return mask

def orientbounds(bounds):
    ret=(900-bounds[2],900-bounds[3],900-bounds[0],900-bounds[1])
    return ret

int_coords = lambda x: np.array(x).round().astype(np.int32)

def polyCoors2mask(poly_coords, shape=(900,900)):
    mask = np.zeros(shape).astype('uint16')
    cv2.fillPoly(mask, [poly_coords], 1)
    mask = mask.astype('uint16')
    return mask

def tilemask_border(tilename, train_builds, build_dist=5):
    kernel = np.ones((build_dist,build_dist),np.uint8)

    temp=train_builds[train_builds.ImageId==tilename]
    temp.reset_index(inplace=True)
    mask = np.zeros((900,900)).astype('uint16')
    borders = np.zeros((900,900)).astype('uint16')
    buildingsG80=0

    if temp.PolygonWKT_Pix.values[0]!='POLYGON EMPTY':
        for i in range(len(temp)):
            P = shapely.wkt.loads(temp.PolygonWKT_Pix.values[i])

            coords=int_coords(P.exterior.coords)
            oneB1 = polyCoors2mask(coords)
            oneB = cv2.dilate(oneB1,kernel)
    #         oneB=oneB-oneB1
            if oneB1.sum()>80:
                buildingsG80 +=1
            mask+=oneB1
            borders+=oneB-oneB1
        mask[mask>0]=1
        borders[ borders>0]=1
        mask = np.stack((mask,borders),-1)
    else:
        mask=np.zeros((900,900,2))
    mask=mask.astype('uint8')
    return mask,  len(temp), buildingsG80

class BuildingsDatasetBordersM(Dataset):
    def __init__(self,
                 datatype: str = 'train',
                 data_folder = '',
                 allimages=None,
                 allmasks=None,
                 sample_weights=None,
                 img_ids: np.array = None,
                 transforms=None,
                 preprocessing=None,
                 get_sample_weight=False):

        self.data_folder = data_folder
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.get_sample_weight = get_sample_weight
        self.datatype = datatype
        self.allimages  = allimages
        self.allmasks = allmasks
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]

        if self.datatype != 'test':
            #

            img = self.allimages[idx]
            #             orientvar=orient_df.orient.loc[orient_df.date==('_').join(image_name.split('_')[:2])].values[0]
            #             mask, buildings, buildingsG80 = tilemask_border(image_name)
            # #             mask, buildings, buildingsG80 = tilemask_pxweight(image_name)

            #             if orientvar==1:
            #                 mask=np.fliplr(np.flipud(mask))
            # #             mask = np.expand_dims(mask,-1)
            mask = self.allmasks[idx]
            buildingsG80 = self.sample_weights[idx]

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']

            if self.get_sample_weight == True:
                return img, mask, buildingsG80 + 1
            else:
                return img, mask

        else:
            image_path = self.data_folder + image_name + '.jpg'
            img = cv2.imread(image_path)
            augmented = self.transforms(image=img, mask=np.zeros_like(img).astype('uint8'))
            img = augmented['image']
            mask = augmented['mask']

            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=np.zeros_like(img).astype('uint8'))
                img = preprocessed['image']
                mask = augmented['mask']

            return img, mask

    def __len__(self):
        return len(self.img_ids)


def cosine_anneal_schedule(t, lr_0, epochs, snapshots):
    cos_inner = np.pi * (t % (epochs // snapshots))
    cos_inner /= epochs // snapshots
    cos_out = np.cos(cos_inner) + 1
    return float(lr_0 / 2 * cos_out)




# cosine_anneal_schedule_list

#def adjust_optim(optimizer, n_iter, epochs):
#    optimizer.param_groups[0]['lr'] = cosine_anneal_schedule_list[n_iter % epochs]


# adjust_optim(optimizer,47)


def train_model(model_encoder='efficientnet-b4',
                bs=6,
                epochs=100,
                train_ids=[],
                output_dir='',
                images_dir='',
                masks_dir='',
                name='exp_name',
                SIZE=480,
                init_lr=0.0001,
                init_model=None,
                output_model=None,
                output_txt=None,
                snapshots=10,
                allimages=None,
                allmasks=None,
                sample_weights=None,
                mode='train'
                ):

    text_name = output_txt

    #train_ids = train_images_with_mask[:]

    print(train_ids[:2])
    print(len(train_ids))
    print(model_encoder, bs, epochs, snapshots, init_lr, SIZE)
    print(output_dir, images_dir, masks_dir, init_model)
    print('in mode ', mode)

    best_val_loss = 10000  # arbitrary large

    ACTIVATION = 'sigmoid'  # None#'sigmoid'
    ENCODER = model_encoder
    ENCODER_WEIGHTS = 'imagenet'
    # DEVICE = 'cpu'
    DEVICE = 'cuda'

    import segmentation_models_pytorch as smp
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=2,
        activation=ACTIVATION,
        decoder_attention_type='scse'
    )

    if init_model is not None:
        if os.path.exists(init_model):
            print('loading model from (2)', init_model)
            model.load_state_dict(torch.load(init_model))

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    cosine_anneal_schedule_list = [cosine_anneal_schedule(x, init_lr, epochs, snapshots) for x in range(epochs)]

    model.cuda()

    # 7955 MB GPU, 20 h one 1 GPU (other model run in parallel on other GPU)
    num_workers = 4

    train_dataset = BuildingsDatasetBordersM(datatype='train', img_ids=train_ids,
                                             transforms=get_training_augmentation(SIZE),
                                             preprocessing=get_preprocessing(preprocessing_fn),
                                             get_sample_weight=True,
                                             data_folder=output_dir,
                                             allimages=allimages,
                                             allmasks=allmasks,
                                             sample_weights=sample_weights
                                             )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    criterion1 = nn.BCELoss(reduction='none')
    with open(text_name, "w") as text_file:
        text_file.write("")

    #mb = master_bar(range(epochs))
    loss_log = []
    for epoch in tqdm(range(epochs)):
        avg_train_loss = 0.
        if mode == 'train':
            model.train()
        else:
            model.eval()
        #pb = progress_bar(train_loader)
        for ii, (data, target, sample_weight) in enumerate(tqdm(train_loader)):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)

            loss1 = criterion1(output[:, 0, ...], target[:, 0, ...].to(dtype=torch.float))
            loss2 = criterion1(output[:, 1, ...], target[:, 1, ...].to(dtype=torch.float))

            loss1 = torch.mean((loss1.mean((1, 2)) * sample_weight.cuda().to(dtype=torch.float)))
            loss2 = torch.mean((loss2.mean((1, 2)) * sample_weight.cuda().to(dtype=torch.float)))
            loss = (loss1 + loss2) / 2

            loss.backward()
            optimizer.step()
            if ii % 1000 == 0:
                loss_log.append(loss.item())
            avg_train_loss += loss.item() / len(train_loader)

        #print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, avg_train_loss))

        model.eval()
        avg_val_loss = 0.

        # Create log
        appendlogfile = 'Epoch: ' + str(epoch) + ' - Loss: ' + str(avg_train_loss) + ' - LR: ' + str(cosine_anneal_schedule_list[(epoch + 1) % epochs]) +  '\n'
        with open(text_name, "a") as text_file:
            text_file.write(appendlogfile)

        #     if avg_val_loss < best_val_loss:
        #         torch.save(model.state_dict(), weights_path+name+'.pt')

        #
        # if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), output_model)

        #     scheduler.step()
        #adjust_optim(optimizer, (epoch + 1))
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule_list[(epoch + 1) % epochs]
        gc.collect()

    with open(text_name, "a") as text_file:
        text_file.write('\n')

    del train_dataset
    del train_loader
    del model
    gc.collect()
    return output_model


class BuildingsDatasetBorders(Dataset):
    def __init__(self, datatype: str = 'train',
                 data_folder = '',
                 img_ids: np.array = None,
                 transforms=None,
                 preprocessing=None,
                 get_sample_weight=False,
                 orient_df=None):
        self.data_folder = data_folder
        # if datatype != 'test':
        #     self.data_folder = '/root/spacenet/train_sar_productscale_orient/'
        # else:
        #     self.data_folder = '/root/spacenet/test_sar_productscale_orient/'
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.get_sample_weight = get_sample_weight
        self.datatype = datatype
        self.orient_df = orient_df

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        image_path = self.data_folder + image_name + '.jpg'
        img = cv2.imread(image_path)
        if self.datatype != 'test':
            orientvar = self.orient_df.orient.loc[self.orient_df.date == ('_').join(image_name.split('_')[:2])].values[0]
            mask, buildings, buildingsG80 = tilemask_border(image_name)
            #             mask, buildings, buildingsG80 = tilemask_pxweight(image_name)

            if orientvar == 1:
                mask = np.fliplr(np.flipud(mask))
            #             mask = np.expand_dims(mask,-1)

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']

            if self.get_sample_weight == True:
                return img, mask, buildingsG80 + 1
            else:
                return img, mask

        else:
            augmented = self.transforms(image=img, mask=np.zeros_like(img).astype('uint8'))
            img = augmented['image']
            mask = augmented['mask']

            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=np.zeros_like(img).astype('uint8'))
                img = preprocessed['image']
                mask = augmented['mask']

            return img, mask

    def __len__(self):
        return len(self.img_ids)
