import argparse
import os

from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import skimage
import skimage.io


from multiprocessing.pool import Pool

import numpy as np
import cv2
cv2.setNumThreads(0)


def average_strategy(images):
    return np.average(images, axis=0)


def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.


def ensemble_image(params):
    file, dirs, ensembling_dir, strategy = params
    images = []
    for dir in dirs:
        file_path = os.path.join(dir, file)
        im = cv2.imread(file_path, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (900, 900))
        images.append(im)
    images = np.array(images).astype(np.float32)

    if strategy == 'average':
        ensembled = average_strategy(images)
    elif strategy == 'hard_voting':
        ensembled = hard_voting(images)
    else:
        raise ValueError('Unknown ensembling strategy')
    cv2.imwrite(os.path.join(ensembling_dir, file), ensembled)


def ensemble(dirs, strategy, ensembling_dir, n_threads):
    files = os.listdir(dirs[0])
    params = []

    for file in files:
        params.append((file, dirs, ensembling_dir, strategy))
    with Pool(n_threads) as pool:
        with tqdm(total=len(params)) as pbar:
            for _ in pool.imap_unordered(ensemble_image, params):
                pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Ensemble masks")
    arg = parser.add_argument
    arg('--ensembling_cpu_threads', type=int, default=12)
    arg('--ensembling_dir', type=str, default='../test_results/ensemble_multiscale')
    arg('--strategy', type=str, default='average')
    arg('--folds_dir', type=str, default='../test_results/multiscale')
    args = parser.parse_args()

    folds_dir = args.folds_dir
    dirs = [os.path.join(folds_dir, d) for d in os.listdir(folds_dir)]
    print(list(os.listdir(folds_dir)))
    for d in dirs:
        if  not os.path.exists(d):
            raise ValueError(d + " doesn't exist")
    os.makedirs(args.ensembling_dir, exist_ok=True)
    ensemble(dirs, args.strategy, args.ensembling_dir, args.ensembling_cpu_threads)
