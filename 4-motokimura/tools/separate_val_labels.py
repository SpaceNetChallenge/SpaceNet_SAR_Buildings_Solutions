#!/usr/bin/env python3
# this script extracts buildings from given ground-truth csv 
# and dumps to different csv files by every val split.
# this is a helper script for SpaceNet official visualizer.
# supposed to be used together with separate_val_images.py.

import argparse
import json
import os
import pandas as pd
import timeit

from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label_csv',
        help='path to building label csv file',
        default='/data/spacenet6/spacenet6/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv'
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
        help='root directory to which split label csv is saved',
        default='/data/spacenet6/val_labels/'
    )
    return parser.parse_args()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    val_split_paths = glob(
        os.path.join(args.split_dir, 'val_*.json')
    )
    assert len(val_split_paths) == args.split_num
    val_split_paths.sort()

    # load ground truth csv
    df = pd.read_csv(args.label_csv)

    # prepare output directory
    os.makedirs(args.out_dir, exist_ok=False)

    for val_split_path in val_split_paths:
        # get image ids included in this split (e.g., "20190822070610_20190822070846_tile_3721")
        with open(val_split_path) as f:
            val_list = json.load(f)

        image_ids = []
        for data in val_list:
            sar_filename = data['SAR-Intensity']
            image_id = '_'.join(os.path.splitext(sar_filename)[0].split('_')[-4:])
            image_ids.append(image_id)

        # extract buildings included in this split
        mask = (df['ImageId'].isin(image_ids))
        df_ = df[mask]

        # dump to csv
        val_split_name, _ = os.path.splitext(os.path.basename(val_split_path))  # supposed to be "val_x"
        out_csv_path = os.path.join(args.out_dir, f'{val_split_name}.csv')
        df_.to_csv(out_csv_path, index=False)

        print(f'successfully saved to {out_csv_path}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
