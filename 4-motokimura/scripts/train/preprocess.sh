#!/bin/bash

source activate solaris

TRAIN_DIR=$1  # path/to/spacenet6/train/AOI_11_Rotterdam/
source /work/settings.sh

echo ''
echo 'computing mean and std of SAR-Intensity images...'
/work/tools/compute_mean_std.py \
    --data_dir ${TRAIN_DIR} \
    --image_subdir SAR-Intensity \
    --out_dir ${IMAGE_MEAN_STD_DIR}

echo ''
echo 'computing mean and std of PS-RGBNIR images...'
/work/tools/compute_mean_std.py \
    --data_dir ${TRAIN_DIR} \
    --image_subdir PS-RGBNIR \
    --out_dir ${IMAGE_MEAN_STD_DIR}

echo ''
echo 'generating building mask images...'
/work/tools/geojson_to_mask.py \
    --data_dir ${TRAIN_DIR} \
    --out_dir ${BUILDING_MASK_DIR} \
    --boundary_width ${BUILDING_BOUNDARY_WIDTH} \
    --min_area ${BUILDING_MIN_AREA}

echo ''
echo 'splitting dataset...'
/work/tools/split_dataset.py \
    --data_dir ${TRAIN_DIR} \
    --mask_dir ${BUILDING_MASK_DIR} \
    --out_dir ${DATA_SPLIT_DIR} \
    --split_num ${DATA_SPLIT_NUM}
