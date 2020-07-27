import argparse
import os
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
import skimage.io
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import remove_small_objects
from tqdm import tqdm


def process_image(img_path,img_dir, water_dir):
    img = skimage.io.imread(os.path.join(img_dir, img_path))

    nir_band = img[..., -1].copy()

    nir_band[nir_band > 70] = 100
    nir_band[nir_band < 70] = 1
    nir_band[nir_band == 100] = 0
    nir_band[np.all(img == 0, axis=-1)] = 0
    nir_band = binary_dilation(nir_band, iterations=2)
    nir_band = binary_erosion(nir_band, iterations=2)
    nir_band = (remove_small_objects(nir_band, 20000) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(water_dir, img_path.replace("SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_", "")[:-4] + ".png"), nir_band)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/PS-RGBNIR/")
    parser.add_argument('--water-dir', default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/water_masks/")

    args = parser.parse_args()
    os.makedirs(args.water_dir, exist_ok=True)

    image_list = os.listdir(args.data_dir)
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(image_list)) as pbar:
            for _ in pool.imap_unordered(partial(process_image, water_dir=args.water_dir, img_dir=args.data_dir),
                                         image_list):
                pbar.update()
