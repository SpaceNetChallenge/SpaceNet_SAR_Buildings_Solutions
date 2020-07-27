import os
import subprocess
from functools import partial
from multiprocessing.pool import Pool
from os import cpu_count
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import skimage.io

from generate_polygons import polygonize
from tools.instance_label import label_mask


def calc_score(labels, y_pred):
    if y_pred.sum() == 0 and labels.sum() == 0:
        return 1
    if labels.sum() == 0 and y_pred.sum() > 0 or y_pred.sum() == 0 and labels.sum() > 0:
        return 0

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    return precision_at(0.5, iou)


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def score(y, p):
    return calc_score(np.expand_dims(y, 0), np.expand_dims(p, 0))


def label_save_mask(tile, preds_dir, labels_dir, orientations):
    pred = cv2.imread(os.path.join(preds_dir, tile))
    orientation = orientations["_".join(tile[:-4].split("_")[-4:-2])]
    pred_labels = label_mask(pred / 255., main_threshold=0.3, seed_threshold=0.6)
    if orientation > 0:
        pred_labels = cv2.rotate(pred_labels, cv2.ROTATE_180)
    pred_labels = pred_labels[14:-14, 14:-14]
    cv2.imwrite(os.path.join(labels_dir, tile[:-4] + ".tiff"), pred_labels.astype(np.uint16))


def calculate_visualizer(visualizer_path, truth_csv, pred_path, img_dir):
    truth_file = truth_csv
    poly_file = pred_path

    cmd = [
        'java',
        '-jar',
        visualizer_path,
        '-truth',
        truth_file,
        '-solution',
        poly_file,
        '-no-gui',
        '-data-dir',
        img_dir
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate()
    lines = [line for line in stdout_data.decode('utf8').strip().split('\n')]
    overall = 0
    for i, line in enumerate(lines):
        if "Overall F-score :" in line:
            overall = float(line.split(":")[-1].strip())
            for l in lines[i: i + 9]:
                print(l)

    return overall


def calculate_metrics(visualizer_path, truth_csv, img_dir, sar_orientations_csv, fold_dir):
    labels_dir = os.path.join(fold_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    preds_dir = os.path.join(fold_dir, "predictions")
    preds_csv_path = os.path.join(fold_dir, "polygons.csv")

    orientations = pd.read_csv(os.path.join(sar_orientations_csv), header=None).values
    orientations_dict = {}
    for row in orientations:
        id, o = row[0].split(" ")
        orientations_dict[id] = float(o)
    files = [f for f in os.listdir(preds_dir) if f.endswith("png")]
    tiles = set([f[:-4] for f in os.listdir(preds_dir) if f.endswith("png")])
    filtered_truth = os.path.join(fold_dir, "truth.csv")
    df = pd.read_csv(truth_csv)
    df[df["ImageId"].isin(tiles)].to_csv(filtered_truth, index=False)
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(files)) as pbar:
            for v in pool.imap_unordered(partial(label_save_mask, preds_dir=preds_dir, labels_dir=labels_dir,
                                                 orientations=orientations_dict),
                                         files):
                pbar.update()
    polygonize(labels_dir, preds_csv_path, is_labels=True)
    return calculate_visualizer(visualizer_path, filtered_truth, preds_csv_path, img_dir)


if __name__ == '__main__':
    score = calculate_metrics(
        visualizer_path="visualizer/visualizer.jar",
        truth_csv="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv",
        img_dir="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/SAR-Intensity/",
        fold_dir="../oof_preds/localization_resnext_unet_resnext101_0/",
        sar_orientations_csv="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt",
    )
    print(score)
