#!/usr/bin/env python3
# this script splits train dataset into train and val splits

import argparse
import json
import os
import timeit

from glob import glob
from tqdm import tqdm

from _init_path import init_path
init_path()

from spacenet6_model.utils import train_list_filename, val_list_filename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='directory containing spacenet6 train dataset',
        default='/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/'
    )
    parser.add_argument(
        '--mask_dir',
        help='directory containing building mask image files',
        default='/data/spacenet6/footprint_boundary_mask/v_01/labels/'
    )
    parser.add_argument(
        '--out_dir',
        help='output root directory',
        default='/data/spacenet6/split/'
    )
    parser.add_argument(
        '--split_num',
        help='number of split',
        type=int,
        default=5
    )
    return parser.parse_args()


def check_filename_validity(data_list):
    N = len(data_list)
    for i in range(N):
        data = data_list[i]

        polygon_filename = data['geojson_buildings']
        ms_image_filename = data['RGBNIR']
        pan_image_filename = data['PAN']
        psms_image_filename = data['PS-RGBNIR']
        rgb_image_filename = data['PS-RGB']
        sar_image_filename = data['SAR-Intensity']
        mask_filename = data['Mask']

        # check if the filenames are valid
        assert polygon_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_Buildings_'
        assert polygon_filename[-8:] == '.geojson'

        assert ms_image_filename[:34] == 'SN6_Train_AOI_11_Rotterdam_RGBNIR_'
        assert ms_image_filename[-4:] == '.tif'

        assert pan_image_filename[:31] == 'SN6_Train_AOI_11_Rotterdam_PAN_'
        assert pan_image_filename[-4:] == '.tif'

        assert psms_image_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_'
        assert psms_image_filename[-4:] == '.tif'

        assert rgb_image_filename[:34] == 'SN6_Train_AOI_11_Rotterdam_PS-RGB_'
        assert rgb_image_filename[-4:] == '.tif'

        assert sar_image_filename[:41] == 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'
        assert sar_image_filename[-4:] == '.tif'

        assert mask_filename[:37] == 'SN6_Train_AOI_11_Rotterdam_Buildings_'
        assert mask_filename[-12:] == '.geojson.png'

        # check if ids match
        identity = polygon_filename[37:-8]
        assert ms_image_filename[34:-4] == identity
        assert pan_image_filename[31:-4] == identity
        assert psms_image_filename[37:-4] == identity
        assert rgb_image_filename[34:-4] == identity
        assert sar_image_filename[41:-4] == identity
        assert mask_filename[37:-12] == identity


def assign_split_id(data_list, sar_image_dir, split_num):
    """Assign the tile to one of a small number of groups (split),
    for setting aside validation data (or for k-fold cross-validation).
    Caveats: These groups slightly overlap each other.
    Also, they are not of equal size.
    """
    import gdal
    import math

    west_edge = 591550  # approximate west edge of training data area
    east_edge = 596250  # approximate east edge of training data area
    
    split_ids = []
    for data in tqdm(data_list):
        sar_path = os.path.join(sar_image_dir, data['SAR-Intensity'])
        sar_gdal_data = gdal.Open(sar_path)
        sar_transform = sar_gdal_data.GetGeoTransform()
        x = sar_transform[0]
        split_id = min(
            split_num - 1,
            max(0, math.floor((x - west_edge) / (east_edge - west_edge) * split_num))
        )
        split_ids.append(split_id)
    return split_ids


def dump_to_files(out_dir, data_list, split_ids, split_num):
    def dump_to_file(out_path, data_list):
        with open(out_path, 'w') as f:
            json.dump(
                data_list,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=False,
                separators=(',', ': ')
            )

    for val_split_id in range(split_num):
        train_list, val_list = [], []
        for data, split_id in zip(data_list, split_ids):
            if split_id == val_split_id:
                val_list.append(data)
            else:
                train_list.append(data)

        dump_to_file(os.path.join(out_dir, train_list_filename(val_split_id)), train_list)
        dump_to_file(os.path.join(out_dir, val_list_filename(val_split_id)), val_list)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Building polygon
    polygon_dir = os.path.join(args.data_dir, 'geojson_buildings')
    polygon_paths = glob(os.path.join(polygon_dir, '*.geojson'))
    polygon_paths.sort()

    # Multi-spectral
    ms_image_dir = os.path.join(args.data_dir, 'RGBNIR')
    ms_image_paths = glob(os.path.join(ms_image_dir, '*.tif'))
    ms_image_paths.sort()

    # Panchromatic
    pan_image_dir = os.path.join(args.data_dir, 'PAN')
    pan_image_paths = glob(os.path.join(pan_image_dir, '*.tif'))
    pan_image_paths.sort()

    # Pan-sharpened Multi-spectral
    psms_image_dir = os.path.join(args.data_dir, 'PS-RGBNIR')
    psms_image_paths = glob(os.path.join(psms_image_dir, '*.tif'))
    psms_image_paths.sort()

    # Pan-sharpened RGB
    rgb_image_dir = os.path.join(args.data_dir, 'PS-RGB')
    rgb_image_paths = glob(os.path.join(rgb_image_dir, '*.tif'))
    rgb_image_paths.sort()

    # SAR intensity
    sar_image_dir = os.path.join(args.data_dir, 'SAR-Intensity')
    sar_image_paths = glob(os.path.join(sar_image_dir, '*.tif'))
    sar_image_paths.sort()

    # Building mask (generated by geojson_to_mask.py)
    mask_paths = glob(os.path.join(args.mask_dir, '*.geojson.png'))
    mask_paths.sort()

    #N = 3401
    #assert len(polygon_paths) == N
    #assert len(ms_image_paths) == N
    #assert len(pan_image_paths) == N
    #assert len(psms_image_paths) == N
    #assert len(rgb_image_paths) == N
    #assert len(sar_image_paths) == N
    #assert len(mask_paths) == N

    N = len(polygon_paths)

    data_list = []
    for i in range(N):
        polygon_filename = os.path.basename(polygon_paths[i])
        ms_image_filename = os.path.basename(ms_image_paths[i])
        pan_image_filename = os.path.basename(pan_image_paths[i])
        psms_image_filename = os.path.basename(psms_image_paths[i])
        rgb_image_filename = os.path.basename(rgb_image_paths[i])
        sar_image_filename = os.path.basename(sar_image_paths[i])
        mask_filename = os.path.basename(mask_paths[i])

        data_list.append(
            {
                'geojson_buildings': polygon_filename,
                'RGBNIR': ms_image_filename,
                'PAN': pan_image_filename,
                'PS-RGBNIR': psms_image_filename,
                'PS-RGB': rgb_image_filename,
                'SAR-Intensity': sar_image_filename,
                'Mask': mask_filename
            }
        )

    # check order of data_list
    #check_filename_validity(data_list)

    # assign split id to each tile based on x-coord
    split_ids = assign_split_id(data_list, sar_image_dir, args.split_num)

    # dump train/val filenames for each split 
    dump_to_files(args.out_dir, data_list, split_ids, args.split_num)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
