#!/usr/bin/env python3
# this script generates building footprint and boundary mask PNG files
# from given building polygon GeoJSON files
# see also notebooks/geojson_to_mask.ipynb

import argparse
import cv2
import geopandas as gpd
import numpy as np
import os
import solaris as sol
import timeit

from glob import glob
from skimage import io
from tqdm import tqdm

from _init_path import init_path
init_path()

from spacenet6_model.utils import get_roi_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='directory containing spacenet6 train dataset',
        default='/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/'
    )
    parser.add_argument(
        '--out_dir',
        help='directory to output mask images',
        default='/data/spacenet6/footprint_boundary_mask/v_01/labels/'
    )
    parser.add_argument(
        '--vis_dir',
        help='directory to output colorized mask images',
        default=''
    )
    parser.add_argument(
        '--boundary_width',
        help='width of building boundary class in pixel',
        type=int,
        default=6
    )
    parser.add_argument(
        '--min_area',
        help='minmum area of building to consider in pixel',
        type=int,
        default=20
    )
    return parser.parse_args()


def check_filenames_validity(sar_image_filename, building_label_filename):
    # check if the filenames are valid
    assert sar_image_filename[:41] == 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'
    assert sar_image_filename[-4:] == '.tif'

    assert building_label_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_Buildings_'
    assert building_label_filename[-8:] == '.geojson'

    # check if both of sar image and building label cover the same tile
    assert sar_image_filename[41:-4] == building_label_filename[37:-8]


def compute_instance_pixel_extents(instance_mask):
    instance_extents = []
    _, _, n_building = instance_mask.shape
    for i in range(n_building):
        extent = (instance_mask[:, :, i] > 0).sum()
        instance_extents.append(extent)
    return instance_extents


def remove_small_building(instance_mask, min_extent):
    instance_extents = compute_instance_pixel_extents(instance_mask)
    return instance_mask[:, :, np.array(instance_extents) > min_extent]


def generate_footprint_mask(instance_mask):
    footprint_mask = (instance_mask.sum(axis=2)) > 0
    footprint_mask = footprint_mask.astype(np.uint8)
    return footprint_mask


def generate_boundary_mask(instance_mask, boundary_width_pixel):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_width_pixel, boundary_width_pixel)
    )
    h, w, n_building = instance_mask.shape
    boundary_mask = np.zeros(shape=[h, w], dtype=np.uint8)
    
    for i in range(n_building):
        dilated = cv2.dilate(
            instance_mask[:, :, i],
            kernel,
            iterations=1
        )
        boundary = dilated - instance_mask[:, :, i]
        boundary_mask[boundary > 0] = 1
        
    return boundary_mask


def combine_masks(footprint_mask, boundary_mask):
    h, w = footprint_mask.shape
    combined_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    combined_mask[footprint_mask > 0] = 1  # 1 for footprint
    combined_mask[boundary_mask > 0] = 2  # 2 for boundary

    combined_mask_color = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    combined_mask_color[combined_mask == 1] = np.array([255, 0, 0])  # red for footprint
    combined_mask_color[combined_mask == 2] = np.array([0, 255, 0])  # green for boundary

    return (
        combined_mask, ## 0: background, 1: footprint, 2: boundary
        combined_mask_color  # black: background, red: footprint, green: boundary
    )


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    os.makedirs(os.path.join(args.out_dir), exist_ok=True)
    if args.vis_dir:
        os.makedirs(os.path.join(args.vis_dir), exist_ok=True)

    # SAR intensity
    sar_image_dir = os.path.join(args.data_dir, 'SAR-Intensity')
    sar_image_paths = glob(os.path.join(sar_image_dir, '*.tif'))
    sar_image_paths.sort()

    # building label
    building_label_dir = os.path.join(args.data_dir, 'geojson_buildings')
    building_label_paths = glob(os.path.join(building_label_dir, '*.geojson'))
    building_label_paths.sort()

    #N = 3401
    #assert len(building_label_paths) == N
    #assert len(sar_image_paths) == N

    N = len(building_label_paths)

    for i in tqdm(range(N)):
        sar_image_path = sar_image_paths[i]
        building_label_path = building_label_paths[i]

        sar_image_filename = os.path.basename(sar_image_path)
        building_label_filename = os.path.basename(building_label_path)
        #check_filenames_validity(sar_image_filename, building_label_filename)

        # load sar image and compute roi mask from sar intensity values
        sar_image = io.imread(sar_image_path)
        roi_mask = get_roi_mask(sar_image)

        # gen building instance mask /w shape of [h, w, n_building]
        instance_mask = sol.vector.mask.instance_mask(
            df=building_label_path,
            reference_im=sar_image_path
        )
        instance_mask[np.logical_not(roi_mask)] = 0

        if instance_mask.ndim == 2:  # if n_building = 1
            instance_mask = instance_mask[:, :, np.newaxis]

        # remove small buildings
        instance_mask = remove_small_building(
            instance_mask,
            min_extent=args.min_area
        )

        # gen building footprint mask of the roi
        footprint_mask = generate_footprint_mask(instance_mask)

        # gen building boundary mask of the roi
        boundary_mask = generate_boundary_mask(
            instance_mask,
            boundary_width_pixel=args.boundary_width
        )
        boundary_mask[np.logical_not(roi_mask)] = 0

        # combine footprint mask and boundary mask
        combined_mask, combined_mask_color = combine_masks(footprint_mask, boundary_mask)

        # save masks
        out_filename = f'{building_label_filename}.png'
        io.imsave(os.path.join(args.out_dir, out_filename), combined_mask)
        if args.vis_dir:
            io.imsave(os.path.join(args.vis_dir, out_filename), combined_mask_color)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
