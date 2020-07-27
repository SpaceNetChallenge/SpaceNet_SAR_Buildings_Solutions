import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gdal
import sys

sys.path.append('.')
import solaris

#
# Create a Dataset

from albumentations.pytorch.transforms import ToTensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    RandomRotate90,
    RandomBrightnessContrast,
    Normalize, RandomCrop, Blur
)

channel_max = [92.87763, 91.97153, 91.65466, 91.9873]

transform_rgb = {
    'train': Compose(
        [RandomCrop(512, 512, p=1.0),
         HorizontalFlip(p=.5),
         VerticalFlip(p=.5), RandomRotate90(p=0.5),
         RandomBrightnessContrast(p=0.3),
         Blur(p=0.3),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
         ToTensor(),
         ]),
    'valid': Compose(
        [CenterCrop(512, 512),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
         ToTensor()]),
    'inference': Compose(
        [
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensor()])

}

transform_sar = {
    'train': Compose(
        [RandomCrop(512, 512, p=1.0),
         RandomBrightnessContrast(p=0.3),
         Blur(p=0.3),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
         ToTensor(),
         ]),
    'valid': Compose(
        [CenterCrop(512, 512),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
         ToTensor()]),
    'inference': Compose(
        [
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensor()])

}


def prepare_df(spacenet_masks_file, spacenet_orientation_file):
    #
    # Read orientation per date
    orient_df = pd.read_csv(spacenet_orientation_file, header=None, sep=" ")
    orient_df.columns = ['date', 'orient']
    print(len(orient_df))
    orient_df.head()

    #
    # read training masks and tile ids
    train_df = pd.read_csv(spacenet_masks_file)
    train_ids = np.sort(np.unique(train_df.ImageId.values))

    df = pd.DataFrame({'ImageId': train_ids})
    df['date'] = df.apply(lambda row: row.ImageId.split("_")[0] + "_" + row.ImageId.split("_")[1], axis=1)
    df = pd.merge(df, orient_df, on='date')
    print(len(df[df.orient == 1]), len(df[df.orient == 0]))
    df['split'] = 0
    df.head()
    df['tile_id'] = df.apply(lambda row: row.ImageId.split("_")[-1], axis=1)
    df['odd'] = df.apply(lambda row: int(row.tile_id) % 2 == 1, axis=1)

    return df


class SpaceNetRGB(Dataset):
    def __init__(self, df, transformers=None,
                 base_path='train/',
                 orientation=False,
                 preampl='SN6_Train_AOI_11_Rotterdam_PS-RGB_',
                 sar_preampl='SN6_Train_AOI_11_Rotterdam_SAR-Intensity_',
                 lab_preampl='SN6_Train_AOI_11_Rotterdam_Buildings_',
                 min_size=20,
                 return_orientation=False,
                 return_labels=False,
                 channels=1,
                 nsize=13,
                 return_contact=False):
        # stuff
        self.df = df
        self.base_path = base_path
        self.rgb_base_path = os.path.join(self.base_path , 'PS-RGB/')
        self.sar_base_path = os.path.join(self.base_path , 'SAR-Intensity/')
        self.lab_base_path =  os.path.join(self.base_path , 'geojson_buildings/')
        self.transformers = transformers
        self.orientation = orientation
        driver = gdal.GetDriverByName('GTiff')
        self.preampl = preampl
        self.sar_preampl = sar_preampl
        self.lab_preampl = lab_preampl
        self.min_size = min_size
        self.return_orientation = return_orientation
        self.return_labels = return_labels
        self.channels = channels
        self.nsize = nsize
        self.return_contact = return_contact

    def __getitem__(self, index):
        row = self.df.iloc[index]
        srcpath = self.rgb_base_path + '/' + self.preampl + row.ImageId + '.tif'
        sarpath = self.sar_base_path + '/' + self.sar_preampl + row.ImageId + '.tif'
        labpath = self.lab_base_path + '/' + self.lab_preampl + row.ImageId + '.geojson'
        tilefile = gdal.Open(srcpath)
        numbands = tilefile.RasterCount

        #
        # image
        data = []
        for i in range(1, numbands + 1):
            banddata = tilefile.GetRasterBand(i).ReadAsArray()
            if self.orientation == True and row.orient == 1:
                banddata = np.fliplr(np.flipud(banddata))  # 180 deg rotation
            data.append(banddata)
        data = np.stack(data, axis=-1)
        data = data.astype('uint8')

        #
        # mask
        maskdata, mcount = read_raw_mask(row.ImageId, sar_base_path=self.sar_base_path, lab_base_path=self.lab_base_path,
                                    min_size=self.min_size,channels=self.channels, nsize=self.nsize, return_contact=self.return_contact)

        img = data  # Image.fromarray(data)
        mask = maskdata  # Image.fromarray(maskdata)

        if self.orientation == True and row.orient == 1:
            mask = np.fliplr(np.flipud(mask))

        if self.transformers is not None:
            augmented = self.transformers(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if self.return_labels:
            if self.return_orientation:
                return img, mask, mcount, row.ImageId, row.orient
            else:
                return img, mask, mcount, row.ImageId
        else:
            if self.return_orientation:
                return img, mask, mcount, row.orient
            else:
                return img, mask, mcount

    def __len__(self):
        return len(self.df)


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def preproc(band, iband, lee_size, scale_max, orientation, orient):
    if lee_size is not None:
        band = lee_filter(band, int(lee_size))
    if scale_max is not None:
        band = (256 * band / scale_max[iband])
    else:
        band = (256 * (band - np.min(band)) / (np.max(band) - np.min(band)))

    if orientation == True and orient == 1:
        band = np.fliplr(np.flipud(band))  # 180 deg rotation

    return band


# from datagen import preproc
import geopandas as gpd
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def read_raw_sar(imageid, sar_base_path='./SAR-Intensity/',
                 sar_preampl='SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'):
    sarpath = sar_base_path + '/' + sar_preampl + imageid + '.tif'

    datei = imageid.split('_')[0] + '_' + imageid.split('_')[1]

    tilefile = gdal.Open(sarpath)
    numbands = tilefile.RasterCount

    #
    # image
    # Channel 1
    data = []
    banddata1 = tilefile.GetRasterBand(1).ReadAsArray()
    data.append(banddata1)

    # Channel 2
    banddata2 = tilefile.GetRasterBand(2).ReadAsArray()
    data.append(banddata2)
    banddata3 = tilefile.GetRasterBand(3).ReadAsArray()
    data.append(banddata3)

    # Channel 3
    banddata4 = tilefile.GetRasterBand(4).ReadAsArray()
    data.append(banddata4)

    data = np.stack(data, axis=-1)

    return data


def read_raw_mask(imageid, min_size=20, nsize=13, channels=1, return_contact=False,
                  sar_base_path= './SAR-Intensity/',
                  sar_preampl='SN6_Train_AOI_11_Rotterdam_SAR-Intensity_',
                  lab_base_path= './geojson_buildings/',
                  lab_preampl='SN6_Train_AOI_11_Rotterdam_Buildings_'):
    # print ('mask', imageid)
    sarpath = sar_base_path + '/' + sar_preampl + imageid + '.tif'
    labpath = lab_base_path + '/' + lab_preampl + imageid + '.geojson'
    #
    # mask
    gdf = gpd.read_file(labpath)
    cut_count = gdf.area > float(80)
    gdf_count = gdf.loc[cut_count]

    cut = gdf.area > float(min_size)
    gdf = gdf.loc[cut]


    gdf['bid'] = gdf.index + 1
    maskdata_orig = solaris.vector.mask.footprint_mask(df=gdf, reference_im=sarpath)
    maskdata = solaris.vector.mask.footprint_mask(df=gdf, reference_im=sarpath, burn_field='bid')
    mm = np.zeros_like(maskdata)

    if maskdata.sum() > 0:
        for i in range(len(gdf)):
            m = (maskdata == i + 1).astype('uint8')
            m = cv2.dilate(m, np.ones((nsize, nsize), dtype='uint8'))
            mm = mm + m

        mm[mm > 1] = 255
        mm[mm <= 1] = 0
        maskdata = mm.copy()

        maskdata = maskdata_orig - maskdata
        maskdata[maskdata < 0] = 0

    else:
        maskdata = maskdata_orig

    if return_contact:
        final_mask = mm
    else:

        if channels == 1:
            final_mask = maskdata
        else:
            final_mask = np.stack((maskdata, mm), axis=-1)

    return final_mask, len(gdf_count) + 1


def scale_sar_max_dict(data, scale_max=[92, 92, 92, 92], scale_min=[0, 0, 0, 0]):
    for i in range(4):
        data[..., i] = (256 * (data[..., i] - scale_min[i]) / (scale_max[i] - scale_min[i]))
    return data


def lee_sar(data, lee_size=3):
    new_data = []
    for i in range(4):
        new_data.append(lee_filter(data[..., i], int(lee_size)))
    return np.stack(new_data, axis=-1)


def sar2rgb(sar):
    rgb = []
    rgb.append(sar[..., 0])
    rgb.append((sar[..., 1] + sar[..., 2]) / 2)
    rgb.append(sar[..., 3])
    rgb = np.stack(rgb, axis=-1)
    return rgb


def patch_left_right(im1, mask1, im2, mask2):
    r = random.random()
    mid = max(2, int(im1.shape[0] * r))
    img_new = np.zeros_like(im1)
    img_new[:, :mid, :] = im1[:, -mid:, :]
    img_new[:, mid:, :] = im2[:, :-mid, :]
    mask_new = np.zeros_like(mask1)
    mask_new[:, :mid] = mask1[:, -mid:]
    mask_new[:, mid:] = mask2[:, :-mid]

    return img_new, mask_new


def patch_top_down(im1, mask1, im2, mask2):
    r = random.random()
    mid = max(2, int(im1.shape[0] * r))
    img_new = np.zeros_like(im1)
    img_new[:mid, :, :] = im1[-mid:, :, :]
    img_new[mid:, :, :] = im2[:-mid, :, :]
    mask_new = np.zeros_like(mask1)
    mask_new[:mid, :] = mask1[-mid:, :]
    mask_new[mid:, :] = mask2[:-mid, :]

    return img_new, mask_new


def flip_sar(data):
    new_data = []
    for i in range(4):
        new_data.append(np.fliplr(np.flipud(data[..., i])))
    return np.stack(new_data, axis=-1)


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


import cv2

import joblib
import skimage
import random


class SpaceNetSAR2RGBSteroids(Dataset):
    def __init__(self, df, transformers=None,
                 base_path='train/',
                 orientation=False,
                 preampl='SN6_Train_AOI_11_Rotterdam_PS-RGB_',
                 sar_preampl='SN6_Train_AOI_11_Rotterdam_SAR-Intensity_',
                 lab_preampl='SN6_Train_AOI_11_Rotterdam_Buildings_',
                 min_size=20,
                 scale_max=True,
                 return_labels=False,
                 return_orientation=False,
                 lee=0,
                 return_contact=True,
                 valida=False,
                 channels=1, nsize=13,
                 max_dict_name='MAX_DICT_v2',
                 min_dict_name='MIN_DICT_v2',
                 pairs_L='pairs_L',
                 pairs_D='pairs_D'
                 ):
        # stuff
        self.df = df
        self.base_path = base_path
        self.sar_base_path = os.path.join(self.base_path , 'SAR-Intensity/')
        self.lab_base_path =  os.path.join(self.base_path , 'geojson_buildings/')
        self.transformers = transformers
        self.orientation = orientation
        driver = gdal.GetDriverByName('GTiff')
        self.preampl = preampl
        self.sar_preampl = sar_preampl
        self.lab_preampl = lab_preampl
        self.min_size = min_size
        self.scale_max = scale_max
        self.return_labels = return_labels
        self.return_orientation = return_orientation
        self.lee_size = lee
        self.return_contact = return_contact
        self.MAX_DICT_v2 = joblib.load(max_dict_name)
        self.MIN_DICT_v2 = joblib.load(min_dict_name)
        self.pairs_D = joblib.load(pairs_D)
        self.pairs_L = joblib.load(pairs_L)
        self.valida = valida
        self.df_imageids = np.unique(df.ImageId.values)
        self.channels = channels
        self.nsize = nsize
        print('valida', self.valida)
        print('lee size ', self.lee_size)
        print('scale_maxt ', self.scale_max)
        print('min_size ', self.min_size)
        print('channels ', self.channels)
        print('border  ', self.nsize)
        print ('pairs', [pairs_L, pairs_D])
        print ('max dict', [max_dict_name, min_dict_name])

        self.left = 0
        self.down = 0
        print('with nasios min/max scaling')

    def __getitem__(self, index):
        row = self.df.iloc[index]

        #sarpath = self.sar_base_path + '/' + self.sar_preampl + row.ImageId + '.tif'
        #labpath = self.lab_base_path + '/' + self.lab_preampl + row.ImageId + '.geojson'
        dateid = row.ImageId.split('_')[0] + '_' + row.ImageId.split('_')[1]
        sar = read_raw_sar(row.ImageId, sar_base_path=self.sar_base_path, sar_preampl=self.sar_preampl)
        mask, mask_count = read_raw_mask(row.ImageId, min_size=self.min_size, channels=self.channels, nsize=self.nsize,
                                         return_contact=self.return_contact, sar_base_path=self.sar_base_path,
                                         lab_base_path=self.lab_base_path)

        #
        # Compose with prob 50/50
        # Check if the current image has a pair
        has_left = False
        has_down = False

        if len(self.pairs_D[row.ImageId]) > 0 and self.pairs_D[row.ImageId][0] in self.df_imageids:
            has_down = True
        if len(self.pairs_L[row.ImageId]) > 0 and self.pairs_L[row.ImageId][0] in self.df_imageids:
            has_left = True

        if not self.valida:
            p1 = random.random()
            if (has_left or has_down) and p1 > 0.3:
                if has_left and not has_down:  # choose left
                    sar1 = read_raw_sar(self.pairs_L[row.ImageId][0], sar_base_path=self.sar_base_path,
                                        sar_preampl=self.sar_preampl)
                    mask1, _ = read_raw_mask(self.pairs_L[row.ImageId][0], min_size=self.min_size,
                                             channels=self.channels, nsize=self.nsize,
                                             return_contact=self.return_contact, sar_base_path=self.sar_base_path, lab_base_path=self.lab_base_path)
                    sar_final, mask_final = patch_left_right(sar, mask, sar1, mask1)
                    # print ('left')
                    self.left = self.left + 1

                if has_down and not has_left:  # choose down
                    sar1 = read_raw_sar(self.pairs_D[row.ImageId][0],
                                        sar_base_path=self.sar_base_path, sar_preampl=self.sar_preampl)
                    mask1, _ = read_raw_mask(self.pairs_D[row.ImageId][0],
                                             min_size=self.min_size, channels=self.channels, nsize=self.nsize,
                                             return_contact=self.return_contact, sar_base_path=self.sar_base_path, lab_base_path=self.lab_base_path)
                    sar_final, mask_final = patch_top_down(sar, mask, sar1, mask1)
                    # print ('down')
                    self.down = self.down + 1

                if has_down and has_left:
                    p2 = random.random()
                    if p2 > 0.5:  # choose left
                        sar1 = read_raw_sar(self.pairs_L[row.ImageId][0],
                                            sar_base_path=self.sar_base_path, sar_preampl=self.sar_preampl)
                        mask1, _ = read_raw_mask(self.pairs_L[row.ImageId][0],
                                                 min_size=self.min_size, channels=self.channels, nsize=self.nsize,
                                                 return_contact=self.return_contact, sar_base_path=self.sar_base_path, lab_base_path=self.lab_base_path)
                        sar_final, mask_final = patch_left_right(sar, mask, sar1, mask1)
                        # print ('left2')
                        self.left = self.left + 1

                    else:  # choose down
                        sar1 = read_raw_sar(self.pairs_D[row.ImageId][0],
                                            sar_base_path=self.sar_base_path, sar_preampl=self.sar_preampl)
                        mask1, _ = read_raw_mask(self.pairs_D[row.ImageId][0], channels=self.channels, nsize=self.nsize,
                                                 min_size=self.min_size, return_contact=self.return_contact,
                                                 sar_base_path=self.sar_base_path, lab_base_path=self.lab_base_path)
                        sar_final, mask_final = patch_top_down(sar, mask, sar1, mask1)
                        # print ('down2')
                        self.down = self.down + 1

                _, mask_count = skimage.measure.label(mask_final, background=0, connectivity=1, return_num=True)

            else:
                sar_final = sar
                mask_final = mask
        else:
            sar_final = sar
            mask_final = mask

        if self.orientation == True and row.orient == 1:
            sar_final = flip_sar(sar_final)
            mask_final = np.fliplr(np.flipud(mask_final))

        if self.scale_max and dateid in self.MAX_DICT_v2:
            sar_final = scale_sar_max_dict(sar_final, self.MAX_DICT_v2[dateid], self.MIN_DICT_v2[dateid])
        else:
            sar_final = scale_sar_max_dict(sar_final)

        if self.lee_size > 0:
            sar_final = lee_sar(sar_final)

        rgb = sar2rgb(sar_final)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype('uint8')

        if self.transformers is not None:
            augmented = self.transformers(image=rgb, mask=mask_final)
            rgb = augmented['image']
            mask_final = augmented['mask']

        if self.return_labels:
            if self.return_orientation:
                return rgb, mask_final, mask_count, row.ImageId, row.orient
            else:
                return rgb, mask_final, mask_count, row.ImageId
        else:
            if self.return_orientation:
                return rgb, mask_final, mask_count, row.orient
            else:
                return rgb, mask_final, mask_count

    def __len__(self):
        return len(self.df)


class SpaceNetSAR2RGBSteroidsInference(Dataset):
    def __init__(self, df, transformers=None,
                 sar_base_path='./SAR-Intensity/',
                 orientation=False,
                 sar_preampl='SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_',
                 min_size=20,
                 scale_max=True,
                 return_labels=False,
                 return_orientation=False,
                 lee=0,
                 return_contact=True,
                 valida=True,
                 max_dict_name='MAX_DICT_v2',
                 min_dict_name='MIN_DICT_v2',
                 pairs_L='pairs_L',
                 pairs_D='pairs_D'
                 ):
        # stuff
        self.df = df
        self.sar_base_path = sar_base_path
        self.transformers = transformers
        self.orientation = orientation
        driver = gdal.GetDriverByName('GTiff')
        self.sar_preampl = sar_preampl
        self.min_size = min_size
        self.scale_max = scale_max
        self.return_labels = return_labels
        self.return_orientation = return_orientation
        self.lee_size = lee
        self.return_contact = return_contact
        self.MAX_DICT_v2 = joblib.load(max_dict_name)
        self.MIN_DICT_v2 = joblib.load(min_dict_name)
        self.pairs_D = joblib.load(pairs_D)
        self.pairs_L = joblib.load(pairs_L)

        self.valida = valida
        self.df_imageids = np.unique(df.ImageId.values)
        print('lee size ', self.lee_size)
        print('remove_contact ', self.return_contact)
        print('scale_maxt ', self.scale_max)
        print('min_size ', self.min_size)
        print ('pairs', [pairs_L, pairs_D])
        print ('max dict', [max_dict_name, min_dict_name])
        self.left = 0
        self.down = 0

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sarpath = self.sar_base_path + '/' + self.sar_preampl + row.ImageId + '.tif'
        dateid = row.ImageId.split('_')[0] + '_' + row.ImageId.split('_')[1]
        sar = read_raw_sar(row.ImageId, sar_base_path=self.sar_base_path, sar_preampl=self.sar_preampl)

        sar_final = sar

        if self.orientation == True and row.orient == 1:
            sar_final = flip_sar(sar_final)

        if self.scale_max and dateid in self.MAX_DICT_v2:
            sar_final = scale_sar_max_dict(sar_final, self.MAX_DICT_v2[dateid], self.MIN_DICT_v2[dateid])
        else:
            sar_final = scale_sar_max_dict(sar_final)

        if self.lee_size > 0:
            sar_final = lee_sar(sar_final)

        rgb = sar2rgb(sar_final)
        rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype('uint8')

        if self.transformers is not None:
            augmented = self.transformers(image=rgb)
            rgb = augmented['image']

        if self.return_labels:
            if self.return_orientation:
                return rgb, row.ImageId, row.orient
            else:
                return rgb, row.ImageId
        else:
            if self.return_orientation:
                return rgb, row.orient
            else:
                return rgb

    def __len__(self):
        return len(self.df)