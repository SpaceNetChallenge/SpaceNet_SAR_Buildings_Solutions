#!/bin/bash

source activate solaris

TEST_DIR=$1  # path/to/spacenet6/test/AOI_11_Rotterdam/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv
TEST_IMAGE_DIR=${TEST_DIR}/SAR-Intensity

source /work/settings.sh

echo ''
echo 'ensembling model predictions...'
/work/tools/ensemble_models.py \
    INPUT.TEST_IMAGE_DIR ${TEST_IMAGE_DIR} \
    INPUT.CLASSES ${CLASSES} \
    PREDICTION_ROOT ${MODEL_PREDICTION_DIR} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"

echo ''
echo 'generating polygon csv file...'
/work/tools/pred_array_to_poly.py \
    INPUT.CLASSES ${CLASSES} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_DIR} \
    POLY_CSV_ROOT ${POLY_CSV_DIR} \
    BOUNDARY_SUBSTRACT_COEFF ${BOUNDARY_SUBSTRACT_COEFF} \
    METHOD_TO_MAKE_POLYGONS ${METHOD_TO_MAKE_POLYGONS} \
    BUILDING_MIM_AREA_PIXEL ${BUILDING_MIM_AREA_PIXEL} \
    BUILDING_SCORE_THRESH ${BUILDING_SCORE_THRESH} \
    WATERSHED_MAIN_THRESH ${WATERSHED_MAIN_THRESH} \
    WATERSHED_SEED_THRESH ${WATERSHED_SEED_THRESH} \
    WATERSHED_MIN_AREA_PIXEL ${WATERSHED_MIN_AREA_PIXEL} \
    WATERSHED_SEED_MIN_AREA_PIXEL ${WATERSHED_SEED_MIN_AREA_PIXEL} \
    ENSEMBLE_EXP_IDS "[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"

echo ''
echo 'predicting with LGBM models...'
/work/tools/test_lgbm.py \
    --solution_csv ${POLY_CSV_DIR}/exp_0005_0006_0007_0008_0009_0010_0011_0012_0013_0014_0015_0016_0017_0018_0019/solution.csv \
    --imageid ${POLY_CSV_DIR}/exp_0005_0006_0007_0008_0009_0010_0011_0012_0013_0014_0015_0016_0017_0018_0019/imageid.json \
    --pred_dir ${ENSEMBLED_PREDICTION_DIR}/exp_0005_0006_0007_0008_0009_0010_0011_0012_0013_0014_0015_0016_0017_0018_0019 \
    --models \
        ${LGBM_MODEL_DIR}/gbm_model_0.txt \
        ${LGBM_MODEL_DIR}/gbm_model_1.txt \
        ${LGBM_MODEL_DIR}/gbm_model_2.txt \
        ${LGBM_MODEL_DIR}/gbm_model_3.txt \
        ${LGBM_MODEL_DIR}/gbm_model_4.txt \
    --out ${OUTPUT_CSV_PATH} \
    --iou_thresh ${IOU_THRESH} \
    --image_dir ${TEST_IMAGE_DIR} \
    --sar_orientation ${SAR_ORIENTATION_PATH}
