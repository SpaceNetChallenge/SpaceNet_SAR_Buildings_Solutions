#!/usr/bin/env python3
# this script conducts model ensemble by averaging 
# score arrays output by different multiple models

import numpy as np
import os.path
import timeit

from glob import glob
from skimage import io
from tqdm import tqdm

from _init_path import init_path
init_path()

from spacenet6_model.configs import load_config
from spacenet6_model.utils import (
    dump_prediction_to_png, ensemble_subdir, experiment_subdir, 
    get_roi_mask, load_prediction_from_png
)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1
    N = len(config.ENSEMBLE_EXP_IDS)

    sar_image_paths = glob(
        os.path.join(config.INPUT.TEST_IMAGE_DIR, '*.tif')
    )
    sar_image_paths.sort()

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    out_dir = os.path.join(config.ENSEMBLED_PREDICTION_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=False)

    for sar_image_path in tqdm(sar_image_paths):
        sar_image = io.imread(sar_image_path)
        roi_mask = get_roi_mask(sar_image)

        h, w = roi_mask.shape
        assert h == 900 and w == 900
        ensembled_score = np.zeros(shape=[len(config.INPUT.CLASSES), h, w])

        sar_image_filename = os.path.basename(sar_image_path)
        array_filename, _ = os.path.splitext(sar_image_filename)
        array_filename = f'{array_filename}.png'

        for exp_id in config.ENSEMBLE_EXP_IDS:
            exp_subdir = experiment_subdir(exp_id)
            score_array = load_prediction_from_png(
                os.path.join(
                    config.PREDICTION_ROOT,
                    exp_subdir,
                    array_filename
                ),
                n_channels=len(config.INPUT.CLASSES)
            )
            score_array[:, np.logical_not(roi_mask)] = 0
            assert score_array.min() >= 0 and score_array.max() <= 1
            ensembled_score += score_array

        ensembled_score = ensembled_score / N
        assert ensembled_score.min() >= 0 and ensembled_score.max() <= 1
        dump_prediction_to_png(
            os.path.join(out_dir, array_filename),
            ensembled_score
        )

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
