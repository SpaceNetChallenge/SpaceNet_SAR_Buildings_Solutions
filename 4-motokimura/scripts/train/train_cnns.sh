#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source /work/settings.sh

TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${DATA_SPLIT_DIR} \
    INPUT.IMAGE_DIR ${TRAIN_DIR} \
    INPUT.BUILDING_DIR ${BUILDING_MASK_DIR} \
    INPUT.SAR_ORIENTATION ${SAR_ORIENTATION_PATH} \
    INPUT.MEAN_STD_DIR ${IMAGE_MEAN_STD_DIR} \
    LOG_ROOT ${TRAIN_LOG_DIR} \
    WEIGHT_ROOT ${MODEL_WEIGHT_DIR} \
    SAVE_CHECKPOINTS ${SAVE_CHECKPOINTS} \
    DUMP_GIT_INFO ${DUMP_GIT_INFO} \
"
# comment out the line below for debug
#TRAIN_ARGS=${TRAIN_ARGS}" SOLVER.EPOCHS 2 EVAL.EPOCH_TO_START_VAL 1"

mkdir -p ${TRAIN_STDOUT_DIR}

echo ''
echo 'training... (1/5)'

## CONFIG_A_1
nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 0 \
    > ${TRAIN_STDOUT_DIR}/exp_0000.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 1 \
    > ${TRAIN_STDOUT_DIR}/exp_0001.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 2 \
    > ${TRAIN_STDOUT_DIR}/exp_0002.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 3 \
    > ${TRAIN_STDOUT_DIR}/exp_0003.out 2>&1 &

wait

echo 'training... (2/5)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 4 \
    > ${TRAIN_STDOUT_DIR}/exp_0004.out 2>&1 &

## CONFIG_A_2 (finetune CONFIG_A_1 models)
nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0000/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 5 \
    > ${TRAIN_STDOUT_DIR}/exp_0005.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0001/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 6 \
    > ${TRAIN_STDOUT_DIR}/exp_0006.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0002/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 7 \
    > ${TRAIN_STDOUT_DIR}/exp_0007.out 2>&1 &

wait

echo 'training... (3/5)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0003/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 8 \
    > ${TRAIN_STDOUT_DIR}/exp_0008.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_A_2} \
    ${TRAIN_ARGS} \
    MODEL.WEIGHT ${MODEL_WEIGHT_DIR}/exp_0004/model_best.pth \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 9 \
    > ${TRAIN_STDOUT_DIR}/exp_0009.out 2>&1 &

## CONFIG_B_1
nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 10 \
    > ${TRAIN_STDOUT_DIR}/exp_0010.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 11 \
    > ${TRAIN_STDOUT_DIR}/exp_0011.out 2>&1 &

wait

echo 'training... (4/5)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 12 \
    > ${TRAIN_STDOUT_DIR}/exp_0012.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 13 \
    > ${TRAIN_STDOUT_DIR}/exp_0013.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_B_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 14 \
    > ${TRAIN_STDOUT_DIR}/exp_0014.out 2>&1 &

## CONFIG_C_1
nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 15 \
    > ${TRAIN_STDOUT_DIR}/exp_0015.out 2>&1 &

wait

echo 'training... (5/5)'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 16 \
    > ${TRAIN_STDOUT_DIR}/exp_0016.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 17 \
    > ${TRAIN_STDOUT_DIR}/exp_0017.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 18 \
    > ${TRAIN_STDOUT_DIR}/exp_0018.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet6_model.py \
    --config ${CONFIG_C_1} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 19 \
    > ${TRAIN_STDOUT_DIR}/exp_0019.out 2>&1 &

wait

echo 'done training all models!'
