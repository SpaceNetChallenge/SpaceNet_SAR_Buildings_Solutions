import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import torch
import albumentations as albu
from torch.utils.data import Dataset, DataLoader


class SemSegDataset(Dataset):  
    def __init__(
            self, 
            images_dir='/data/SN6_buildings/train/AOI_11_Rotterdam/',
            masks_dir='/wdata/train_masks',
            data_type='SAR-Intensity',
            mode='train',
            folds_file='/wdata/folds.csv',
            fold_number=1,
            n_classes=2,
            augmentation=None, 
            preprocessing=None,
            limit_files=None,
            multiplier=5
            
    ):
        folds = pd.read_csv(folds_file, dtype={'image_name': object})
        if data_type != 'all':
            images_dir = os.path.join(images_dir, data_type)
        if mode == 'train' and fold_number != -1:
            folds = folds[folds['fold_number'] != fold_number]['image_name'].tolist()
        elif mode == 'valid' and fold_number != -1:
            folds = folds[folds['fold_number'] == fold_number]['image_name'].tolist()
        else:
            folds = folds['image_name'].tolist()

        if limit_files:
            folds = folds[:limit_files]
        if multiplier and mode == 'train':
            folds = folds * multiplier

        self.n_classes = n_classes
        self.ids = folds
        self.data_type = data_type
        self.folds = folds

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img
    
    def __getitem__(self, i):
        
        # read data
        if self.data_type == 'SAR-Intensity':
            image_path = os.path.join(self.images_dir, 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_' + self.ids[i] + '.tif')
        elif self.data_type == 'PS-RGB':
            image_path = os.path.join(self.images_dir, 'SN6_Train_AOI_11_Rotterdam_PS-RGB_' + self.ids[i] + '.tif')
        else:
            raise ValueError
        mask_path = os.path.join(self.masks_dir, self.ids[i] + '.tif')

        image = self._read_img(image_path)
        mask = self._read_img(mask_path)[:, :, :self.n_classes]
        if self.augmentation:
            sample = albu.Compose(self.augmentation, p=1)(image=image,
                                                          mask=mask)
            
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            image = self.preprocessing(image)

        mask = mask[...] / 255.0

        image = np.moveaxis(image, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float)
        mask = torch.as_tensor(mask, dtype=torch.float)
        # print(mask[0, ...].sum(), mask[1, ...].sum())
        return image, mask
        
    def __len__(self):
        return len(self.ids)


class TestSemSegDataset(Dataset):
    def __init__(
            self,
            images_dir='/data/SN6_buildings/test_public/AOI_11_Rotterdam/SAR-Intensity/',
            augmentation=None,
            preprocessing=None,
            limit_files=None

    ):
        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        ids = sorted(os.listdir(images_dir))
        if limit_files:
            ids = ids[:limit_files]
        self.ids = ids

    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):

        image_path = os.path.join(self.images_dir, self.ids[i])
        image = self._read_img(image_path)
        if self.augmentation:
            sample = albu.Compose(self.augmentation, p=1)(image=image)

            image = sample['image']

        if self.preprocessing:
            image = self.preprocessing(image)

        image = np.moveaxis(image, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float)
        return image

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    print('Validation dataset testing')
    dataset = SemSegDataset(mode='valid', augmentation=albu.Compose([albu.RandomCrop(320, 320)], p=1))
    check_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=False)
    for step, (x, y) in enumerate(check_loader):
        if step > 10:
            break
        print('step is', step, x.shape, y.shape)

    print('Test dataset testing')
    dataset = TestSemSegDataset(augmentation=albu.Compose([albu.PadIfNeeded(928, 928)], p=1))
    check_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=False)
    for step, x in enumerate(check_loader):
        if step > 10:
            break
        print('step is', step, x.shape)