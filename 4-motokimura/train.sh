#!/bin/bash

START_TIME=$SECONDS

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source /work/settings.sh

# remove trained models
rm -rf ${MODEL_ROOT_DIR}/*

# preprocess dataset
/work/scripts/train/preprocess.sh ${TRAIN_DIR}

# train CNN models
/work/scripts/train/train_cnns.sh ${TRAIN_DIR}

# train LGBM models
/work/scripts/train/train_lgbm.sh ${TRAIN_DIR}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo 'Total time for training: ' $(($ELAPSED_TIME/60+1)) '[min]'
