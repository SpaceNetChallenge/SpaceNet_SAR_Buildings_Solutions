import os

import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset


class SpacenetLocDataset(Dataset):
    def __init__(self, data_path, mode, fold=0, folds_csv='folds.csv', transforms=None, normalize=None, multiplier=1,
                 fix_orientation=True, filter_on_border=True):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.fix_orientation = fix_orientation

        df = pd.read_csv(folds_csv)
        self.df = df
        self.normalize = normalize
        self.fold = fold
        if self.mode == "train":
            ids = df[df['fold'] != fold]['id'].tolist()
        else:
            if filter_on_border:
                ids = list(set(df[(df['fold'] == fold) & (df["onborder"] == False)]['id'].tolist()))
            else:
                ids = list(set(df[(df['fold'] == fold)]['id'].tolist()))
        self.transforms = transforms
        self.names = ids
        if mode == "train":
            self.names = self.names * multiplier
        orientations = pd.read_csv(os.path.join(data_path, "SummaryData/SAR_orientations.txt"), header=None).values
        orientations_dict = {}
        for row in orientations:
            id, o = row[0].split(" ")
            orientations_dict[id] = float(o)
        self.orientations_dict = orientations_dict

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]
        img_path = os.path.join(self.data_path, "SAR-Intensity",
                                "SN6_Train_AOI_11_Rotterdam_SAR-Intensity_" + name + ".tif")
        image = skimage.io.imread(img_path)
        image = (image * (255 / 92)).astype(np.uint8)
        mask_path = os.path.join("/wdata/masks", name + ".png")
        mask = cv2.imread(mask_path)
        # water_mask = cv2.imread(os.path.join(self.data_path, "water_masks", name + ".png"), cv2.IMREAD_GRAYSCALE)
        # water_mask = np.expand_dims(water_mask, -1)
        # mask = np.concatenate([mask, water_mask], -1)
        orientation = self.orientations_dict["_".join(name.split("_")[:2])]

        if orientation > 0 and self.fix_orientation:
            image = cv2.rotate(image, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
        sample = self.transforms(image=image, mask=mask)
        sample['img_name'] = name
        sample['orientation'] = orientation
        sample['mask'] = torch.from_numpy(np.ascontiguousarray(np.moveaxis(sample["mask"], -1, 0))).float() / 255.
        sample['image'] = torch.from_numpy(np.moveaxis(sample["image"] / 255., -1, 0).astype(np.float32))
        return sample


class TestSpacenetLocDataset(Dataset):
    def __init__(self, data_path, transforms, orientation_csv):
        super().__init__()
        self.data_path = data_path
        self.names = [f[:-4] for f in os.listdir(os.path.join(data_path, "SAR-Intensity")) if f.endswith("tif")]
        self.transforms = transforms
        orientations = pd.read_csv(orientation_csv, header=None).values
        orientations_dict = {}
        for row in orientations:
            id, o = row[0].split(" ")
            orientations_dict[id] = float(o)
        self.orientations_dict = orientations_dict


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.data_path, "SAR-Intensity", name + ".tif")
        image = skimage.io.imread(img_path)
        image = (image * (255 / 92)).astype(np.uint8)
        orientation = self.orientations_dict["_".join(name.split("_")[-4:-2])]
        if orientation > 0:
            image = cv2.rotate(image, cv2.ROTATE_180)
        sample = self.transforms(image=image)
        sample['img_name'] = name
        sample['orientation'] = orientation
        sample['image'] = torch.from_numpy(np.moveaxis(sample["image"] / 255., -1, 0).astype(np.float32))
        return sample
