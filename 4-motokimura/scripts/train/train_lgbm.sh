#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source /work/settings.sh

# prepare images
echo ''
echo 'splitting val images...'
/work/tools/separate_val_images.py \
    --data_dir ${TRAIN_DIR} \
    --split_dir ${DATA_SPLIT_DIR} \
    --split_num ${DATA_SPLIT_NUM} \
    --out_dir ${VAL_IMAGE_DIR} \
    --image_types SAR-Intensity

# prepare labels
TRAIN_LABEL_CSV=${TRAIN_DIR}/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv

echo ''
echo 'splitting val labels...'
/work/tools/separate_val_labels.py \
    --label_csv ${TRAIN_LABEL_CSV} \
    --split_dir ${DATA_SPLIT_DIR} \
    --split_num ${DATA_SPLIT_NUM} \
    --out_dir ${VAL_LABEL_DIR}

# predict to val
VAL_ARGS="\
    --exp_log_dir ${TRAIN_LOG_DIR} \
    TEST_TO_VAL True \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    PREDICTION_ROOT ${VAL_PREDICTION_DIR} \
"

mkdir -p ${LGBM_STDOUT_DIR}

echo ''
echo 'predicting to val... (1/4)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 5 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0005.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet6_model.py \
    --exp_id 6 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0006.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet6_model.py \
    --exp_id 7 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0007.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/test_spacenet6_model.py \
    --exp_id 8 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0008.out 2>&1 &

wait

echo 'predicting to val... (2/4)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 9 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0009.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet6_model.py \
    --exp_id 10 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0010.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet6_model.py \
    --exp_id 11 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0011.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/test_spacenet6_model.py \
    --exp_id 12 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0012.out 2>&1 &

wait

echo 'predicting to val... (3/4)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 13 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0013.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet6_model.py \
    --exp_id 14 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0014.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet6_model.py \
    --exp_id 15 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0015.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/test_spacenet6_model.py \
    --exp_id 16 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0012.out 2>&1 &

wait

echo 'predicting to val... (4/4)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet6_model.py \
    --exp_id 17 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0017.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet6_model.py \
    --exp_id 18 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0018.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet6_model.py \
    --exp_id 19 \
    ${VAL_ARGS} \
    > ${LGBM_STDOUT_DIR}/val_0019.out 2>&1 &

wait

# ensemble predictions
ENSEMBLE_ARGS="\
    INPUT.CLASSES ${CLASSES} \
    PREDICTION_ROOT ${VAL_PREDICTION_DIR} \
    ENSEMBLED_PREDICTION_ROOT ${VAL_ENSEMBLED_PREDICTION_DIR} \
"

echo ''
echo 'ensembling val predictions...'

nohup /work/tools/ensemble_models.py \
    ${ENSEMBLE_ARGS} \
    ENSEMBLE_EXP_IDS "[5, 10, 15]" \
    INPUT.TEST_IMAGE_DIR ${VAL_IMAGE_DIR}/val_0/SAR-Intensity \
    > ${LGBM_STDOUT_DIR}/ens_0005_0010_0015.out 2>&1 &

nohup /work/tools/ensemble_models.py \
    ${ENSEMBLE_ARGS} \
    ENSEMBLE_EXP_IDS "[6, 11, 16]" \
    INPUT.TEST_IMAGE_DIR ${VAL_IMAGE_DIR}/val_1/SAR-Intensity \
    > ${LGBM_STDOUT_DIR}/ens_0006_0011_0016.out 2>&1 &

nohup /work/tools/ensemble_models.py \
    ${ENSEMBLE_ARGS} \
    ENSEMBLE_EXP_IDS "[7, 12, 17]" \
    INPUT.TEST_IMAGE_DIR ${VAL_IMAGE_DIR}/val_2/SAR-Intensity \
    > ${LGBM_STDOUT_DIR}/ens_0007_0012_0017.out 2>&1 &

nohup /work/tools/ensemble_models.py \
    ${ENSEMBLE_ARGS} \
    ENSEMBLE_EXP_IDS "[8, 13, 18]" \
    INPUT.TEST_IMAGE_DIR ${VAL_IMAGE_DIR}/val_3/SAR-Intensity \
    > ${LGBM_STDOUT_DIR}/ens_0008_0013_0018.out 2>&1 &

nohup /work/tools/ensemble_models.py \
    ${ENSEMBLE_ARGS} \
    ENSEMBLE_EXP_IDS "[9, 14, 19]" \
    INPUT.TEST_IMAGE_DIR ${VAL_IMAGE_DIR}/val_4/SAR-Intensity \
    > ${LGBM_STDOUT_DIR}/ens_0009_0014_0019.out 2>&1 &

wait

# mask to poly
POLY_ARGS="\
    INPUT.CLASSES ${CLASSES} \
    ENSEMBLED_PREDICTION_ROOT ${VAL_ENSEMBLED_PREDICTION_DIR} \
    POLY_CSV_ROOT ${VAL_POLY_CSV_DIR} \
    BOUNDARY_SUBSTRACT_COEFF ${BOUNDARY_SUBSTRACT_COEFF} \
    METHOD_TO_MAKE_POLYGONS ${METHOD_TO_MAKE_POLYGONS} \
    BUILDING_MIM_AREA_PIXEL ${BUILDING_MIM_AREA_PIXEL} \
    BUILDING_SCORE_THRESH ${BUILDING_SCORE_THRESH} \
    WATERSHED_MAIN_THRESH ${WATERSHED_MAIN_THRESH} \
    WATERSHED_SEED_THRESH ${WATERSHED_SEED_THRESH} \
    WATERSHED_MIN_AREA_PIXEL ${WATERSHED_MIN_AREA_PIXEL} \
    WATERSHED_SEED_MIN_AREA_PIXEL ${WATERSHED_SEED_MIN_AREA_PIXEL}
"

echo ''
echo 'generating val polygon csv files...'

nohup /work/tools/pred_array_to_poly.py \
    ${POLY_ARGS} \
    ENSEMBLE_EXP_IDS "[5, 10, 15]" \
    > ${LGBM_STDOUT_DIR}/poly_0005_0010_0015.out 2>&1 &

nohup /work/tools/pred_array_to_poly.py \
    ${POLY_ARGS} \
    ENSEMBLE_EXP_IDS "[6, 11, 16]" \
    > ${LGBM_STDOUT_DIR}/poly_0006_0011_0016.out 2>&1 &

nohup /work/tools/pred_array_to_poly.py \
    ${POLY_ARGS} \
    ENSEMBLE_EXP_IDS "[7, 12, 17]" \
    > ${LGBM_STDOUT_DIR}/poly_0007_0012_0017.out 2>&1 &

nohup /work/tools/pred_array_to_poly.py \
    ${POLY_ARGS} \
    ENSEMBLE_EXP_IDS "[8, 13, 18]" \
    > ${LGBM_STDOUT_DIR}/poly_0008_0013_0018.out 2>&1 &

nohup /work/tools/pred_array_to_poly.py \
    ${POLY_ARGS} \
    ENSEMBLE_EXP_IDS "[9, 14, 19]" \
    > ${LGBM_STDOUT_DIR}/poly_0009_0014_0019.out 2>&1 &

wait

# train LGBM models
echo ''
echo 'training LGBM models...'
/work/tools/train_lgbm.py \
    --truth_csvs \
        ${VAL_LABEL_DIR}/val_0.csv \
        ${VAL_LABEL_DIR}/val_1.csv \
        ${VAL_LABEL_DIR}/val_2.csv \
        ${VAL_LABEL_DIR}/val_3.csv \
        ${VAL_LABEL_DIR}/val_4.csv \
    --solution_csvs \
        ${VAL_POLY_CSV_DIR}/exp_0005_0010_0015/solution.csv \
        ${VAL_POLY_CSV_DIR}/exp_0006_0011_0016/solution.csv \
        ${VAL_POLY_CSV_DIR}/exp_0007_0012_0017/solution.csv \
        ${VAL_POLY_CSV_DIR}/exp_0008_0013_0018/solution.csv \
        ${VAL_POLY_CSV_DIR}/exp_0009_0014_0019/solution.csv \
    --imageids \
        ${VAL_POLY_CSV_DIR}/exp_0005_0010_0015/imageid.json \
        ${VAL_POLY_CSV_DIR}/exp_0006_0011_0016/imageid.json \
        ${VAL_POLY_CSV_DIR}/exp_0007_0012_0017/imageid.json \
        ${VAL_POLY_CSV_DIR}/exp_0008_0013_0018/imageid.json \
        ${VAL_POLY_CSV_DIR}/exp_0009_0014_0019/imageid.json \
    --pred_dirs \
        ${VAL_ENSEMBLED_PREDICTION_DIR}/exp_0005_0010_0015 \
        ${VAL_ENSEMBLED_PREDICTION_DIR}/exp_0006_0011_0016 \
        ${VAL_ENSEMBLED_PREDICTION_DIR}/exp_0007_0012_0017 \
        ${VAL_ENSEMBLED_PREDICTION_DIR}/exp_0008_0013_0018 \
        ${VAL_ENSEMBLED_PREDICTION_DIR}/exp_0009_0014_0019 \
    --out_dir ${LGBM_MODEL_DIR} \
    --image_dir ${TRAIN_DIR}/SAR-Intensity \
    --sar_orientation ${SAR_ORIENTATION_PATH}

echo 'done training LGBM models!'
