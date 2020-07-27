import argparse
import os
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkt
from tqdm import tqdm

from tools.instance_label import label_mask

MIN_AREA = 100


def mask_to_polygons(image_id, mask_dir, is_labels=False):
    lines = []
    if is_labels:
        img1 = cv2.imread(os.path.join(mask_dir, image_id), cv2.IMREAD_UNCHANGED)
        labels = img1.astype(np.uint16)
    else:
        mask = cv2.imread(os.path.join(mask_dir, image_id), cv2.IMREAD_COLOR)
        labels = label_mask(mask / 255., main_threshold=0.32, seed_threshold=0.65, w_pixel_t=70, pixel_t=150)
    building_id = 0
    for i in range(1, labels.max() + 1):
        df_poly = mask_to_poly(labels == i, min_polygon_area_th=MIN_AREA)
        if len(df_poly) > 0:
            building_id += 1
            for i, row in df_poly.iterrows():
                line = "{},{},\"{}\",{:.6f}\n".format(
                    "_".join(image_id.rstrip(".tiff").rstrip(".png").split("_")[-4:]),
                    building_id,
                    row.wkt,
                    row.area_ratio)
                #line = _remove_interiors(line)
                lines.append(line)
    if building_id == 0:
        line = "{},{},{},0\n".format(
            "_".join(image_id.rstrip(".tiff").rstrip(".png").split("_")[-4:]),
            -1,
            "LINESTRING EMPTY")
        lines.append(line)
    return lines


def polygonize(mask_dir, out_file, is_labels=False):
    all_lines = []
    fn_out = out_file
    test_image_list = os.listdir(os.path.join(mask_dir))
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(test_image_list)) as pbar:
            for lines in pool.imap_unordered(partial(mask_to_polygons, mask_dir=mask_dir, is_labels=is_labels), test_image_list):
                all_lines.extend(lines)
                pbar.update()


    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        for l in all_lines:
            f.write(l)


def mask_to_poly(mask, min_polygon_area_th=MIN_AREA):
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df


def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
                line_prefix +
                '))",' +
                line_terminate
        )
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Postprocessing")
    arg = parser.add_argument
    arg('--masks-path', type=str, default='../test_results/ensemble_multiscale', help='Path to predicted masks')
    arg('--output-path', type=str, help='Path for output file', default="submission.csv")

    args = parser.parse_args()


    polygonize(args.masks_path, args.output_path)
