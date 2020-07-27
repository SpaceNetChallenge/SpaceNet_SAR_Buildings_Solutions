import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count
import cv2
import numpy as np
import pandas as pd
from cv2 import fillPoly
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.morphology import square, dilation, watershed, erosion
from tqdm import tqdm

warnings.simplefilter("ignore")

def create_mask(labels):
    border_msk = np.zeros_like(labels, dtype='bool')
    for l in range(1, labels.max() + 1):
        tmp_lbl = labels == l
        _k = square(4)
        tmp = erosion(tmp_lbl, _k)
        tmp = tmp ^ tmp_lbl
        border_msk = border_msk | tmp

    tmp = dilation(labels > 0, square(9))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = tmp | border_msk
    tmp = dilation(tmp, square(9))

    msk0 = labels > 0

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 3
            else:
                sz = 2
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    msk1 = 255 * msk1
    msk1 = msk1.astype('uint8')

    msk0 = 255 * msk0
    msk0 = msk0.astype('uint8')

    msk2 = 255 * border_msk
    msk2 = msk2.astype('uint8')
    msk = np.stack((msk0, msk2, msk1))
    msk = np.rollaxis(msk, 0, 3)

    return msk


def generate_labels(data, labels_dir, masks_dir):
    tile_id, polygons = data
    labels = np.zeros((900, 900), np.uint16)
    label = 1

    for feat in polygons:
        if feat == "LINESTRING EMPTY" or feat == "POLYGON EMPTY":
            continue
        feat = feat.replace("POLYGON ((", "").replace("), (", "|").replace("),(", "|").replace("(", "").replace(")", "")
        feat_polygons = feat.split("|")
        for i, polygon in enumerate(feat_polygons):
            polygon_coords = []
            for xy in polygon.split(","):
                xy = xy.strip()
                x, y = xy.split(" ")
                x = float(x)
                y = float(y)
                polygon_coords.append([x, y])

            coords = np.around(np.array(polygon_coords)).astype(np.int32)
            fillPoly(labels, [coords], label if i == 0 else 0)
            label += 1
        #labels = remove_small_objects(labels, min_size=50)
    cv2.imwrite(os.path.join(labels_dir, tile_id + ".tiff"), labels)
    cv2.imwrite(os.path.join(masks_dir, tile_id + ".png"), create_mask(labels))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam")
    parser.add_argument('--labels-dir', default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/labels/")
    parser.add_argument('--masks-dir', default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/masks/")

    args = parser.parse_args()
    os.makedirs(args.labels_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)
    csv_path = os.path.join(args.data_path, "SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv")
    df = pd.read_csv(csv_path)
    polygons = df[["ImageId", "PolygonWKT_Pix"]].values
    groups = defaultdict(list)
    for row in polygons:
        tile_id, polygon = row
        groups[tile_id].append(polygon)

    with Pool(cpu_count()) as pool:
        with tqdm(total=len(groups)) as pbar:
            for _ in pool.imap_unordered(partial(generate_labels, labels_dir=args.labels_dir, masks_dir=args.masks_dir),
                                         groups.items()):
                pbar.update()
