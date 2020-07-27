#!/bin/bash

FEATURE_DIR=/wdata
MODEL_ROOT_DIR=/work/models

# path to SAR_orientations.txt
SAR_ORIENTATION_PATH=/work/static/SAR_orientations.txt

# compute_mean_std.py
IMAGE_MEAN_STD_DIR=${MODEL_ROOT_DIR}/image_mean_std

# geojson_to_mask.py
BUILDING_MASK_DIR=${FEATURE_DIR}/masks
BUILDING_BOUNDARY_WIDTH=6
BUILDING_MIN_AREA=20

# split_dataset.py
DATA_SPLIT_DIR=${FEATURE_DIR}/split
DATA_SPLIT_NUM=5

# separate_val_images.py
VAL_IMAGE_DIR=${FEATURE_DIR}/val_images

# separate_val_labels.py
VAL_LABEL_DIR=${FEATURE_DIR}/val_labels

# train_spacenet6_model.py
CONFIG_A_1=/work/configs/deploy/unet-scse_efficientnet-b7_ps-rgbnir_v_01.yml
CONFIG_A_2=/work/configs/deploy/unet-scse_efficientnet-b7_v_01.yml
CONFIG_B_1=/work/configs/deploy/unet-scse_timm-efficientnet-b7_v_01.yml
CONFIG_C_1=/work/configs/deploy/unet-scse_timm-efficientnet-b8_v_01.yml

TRAIN_LOG_DIR=${MODEL_ROOT_DIR}/logs
MODEL_WEIGHT_DIR=${MODEL_ROOT_DIR}/weights
SAVE_CHECKPOINTS='False'
DUMP_GIT_INFO='False'

TRAIN_STDOUT_DIR=/wdata/stdout/train

# train_lgbm.py
LGBM_MODEL_DIR=${MODEL_ROOT_DIR}/gbm_models

LGBM_STDOUT_DIR=/wdata/stdout/lgbm

# test_spacenet6_model.py
MODEL_PREDICTION_DIR=${FEATURE_DIR}/predictions
VAL_PREDICTION_DIR=${FEATURE_DIR}/val_predictions

TEST_STDOUT_DIR=/wdata/stdout/test

# ensemble_models.py
CLASSES='["building_footprint","building_boundary"]'
ENSEMBLED_PREDICTION_DIR=${FEATURE_DIR}/ensembled_predictions
VAL_ENSEMBLED_PREDICTION_DIR=${FEATURE_DIR}/val_ensembled_predictions

# pred_array_to_poly.py
POLY_CSV_DIR=${FEATURE_DIR}/polygons
BOUNDARY_SUBSTRACT_COEFF=0.2
METHOD_TO_MAKE_POLYGONS='watershed'  # ['contours', 'watershed']
BUILDING_MIM_AREA_PIXEL=0  # for 'contours'
BUILDING_SCORE_THRESH=0.5  # for 'contours'
WATERSHED_MAIN_THRESH=0.3  # for 'watershed'
WATERSHED_SEED_THRESH=0.7  # for 'watershed'
WATERSHED_MIN_AREA_PIXEL=80  # for 'watershed'
WATERSHED_SEED_MIN_AREA_PIXEL=20  # for 'watershed'
VAL_POLY_CSV_DIR=${FEATURE_DIR}/val_polygons

# test_lgbm.py
IOU_THRESH=0.1
