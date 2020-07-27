#!/usr/bin/env python3
# this script copies train/val images to different directories by every val split.
# this is a helper script for SpaceNet official visualizer.
# supposed to be used together with separate_val_labels.py.

import argparse
import json
import os
import shutil
import timeit

from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='directory containing spacenet6 train dataset',
        default='/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/'
    )
    parser.add_argument(
        '--split_dir',
        help='directory containing val_x.json files',
        default='/data/spacenet6/split/'
    )
    parser.add_argument(
        '--split_num',
        help='number of split',
        type=int,
        default=5
    )
    parser.add_argument(
        '--out_dir',
        help='root directory to which images are copied',
        default='/data/spacenet6/val_images/'
    )
    parser.add_argument(
        '--image_types',
        help='image types to copy',
        nargs='+',
        default=['SAR-Intensity', 'PS-RGB']
    )
    return parser.parse_args()


def copy_images(val_list_path, data_dir, out_dir, image_type):
    # get full paths to images to copy
    with open(val_split_path) as f:
        val_list = json.load(f)
    image_paths = [
        os.path.join(
            data_dir,
            image_type,
            filenames[image_type]
        ) for filenames in val_list
    ]

    # prepare output directory
    val_split_name, _ = os.path.splitext(os.path.basename(val_split_path))  # supposed to be "val_x"
    out_dir_ = os.path.join(out_dir, val_split_name, image_type)
    os.makedirs(out_dir_, exist_ok=False)

    # copy image files under {args.out_dir}/{val_x}/{image_type}/
    for src_path in tqdm(image_paths):
        shutil.copy(src_path, out_dir_)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    val_split_paths = glob(
        os.path.join(args.split_dir, 'val_*.json')
    )
    assert len(val_split_paths) == args.split_num
    val_split_paths.sort()

    for image_type in args.image_types:
        for val_split_path in val_split_paths:
            val_split_filename = os.path.basename(val_split_path)
            print(f'copying {image_type} files listed by {val_split_filename}...')
            copy_images(val_split_path, args.data_dir, args.out_dir, image_type)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
