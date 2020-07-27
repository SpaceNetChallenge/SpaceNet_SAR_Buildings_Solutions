#!/usr/bin/env bash
ARG1=${1:-/data/SN6_buildings/test_public/AOI_11_Rotterdam/}
ARG2=${2:-/wdata/solution.csv}

mkdir -p /wdata/segmentation_logs/ /wdata/folds_predicts/

if [ "$(ls -A /wdata/segmentation_logs/)" ]; then
     echo "trained weights available"
else
    echo "loading pretrained weights"
    mkdir -p /wdata/segmentation_logs/8_3_reduce_1_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_2_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_3_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_4_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_5_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_6_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_7_unet_senet154/checkpoints/
    mkdir -p /wdata/segmentation_logs/8_3_reduce_8_unet_senet154/checkpoints/
    gdown https://drive.google.com/uc?id=15YTIHPA4hs2tJMFXP0q_2y2B_-UmaQl0 -O /wdata/segmentation_logs/8_3_reduce_1_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=12Qe47uM082NKJ5bqdgmpT_4etTd9a3Q9 -O /wdata/segmentation_logs/8_3_reduce_2_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1z45o0HOGz82YzzcDXBJ56BOsbT97jM3Y -O /wdata/segmentation_logs/8_3_reduce_3_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1NGxMXeXz1L-Z_jGQsyOu53beE9zf6cTg -O /wdata/segmentation_logs/8_3_reduce_4_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1qltahKKpnkxPr2LwDe7MXxt1TrDhSKkc -O /wdata/segmentation_logs/8_3_reduce_5_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=13s1NIqdVQMEHZ4kIMl0y1h5oOhUVQDE_ -O /wdata/segmentation_logs/8_3_reduce_6_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1NeWZAFZLJfhwzWOFragAHyoL0vBUEHxK -O /wdata/segmentation_logs/8_3_reduce_7_unet_senet154/checkpoints/best.pth
    gdown https://drive.google.com/uc?id=1UfYEtvzdIT6lSlp2OaHzjK0oEkl3Mvqi -O /wdata/segmentation_logs/8_3_reduce_8_unet_senet154/checkpoints/best.pth

fi


python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold1.py --gpu '"0"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold2.py --gpu '"1"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold3.py --gpu '"2"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold4.py --gpu '"3"' --test_images $ARG1 --workers 16 --batch_size 16 & wait

python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold5.py --gpu '"0"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold6.py --gpu '"1"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold7.py --gpu '"2"' --test_images $ARG1 --workers 16 --batch_size 16 \
& python3 /project/predict/predict.py --config_path /project/configs/senet154_gcc_fold8.py --gpu '"3"' --test_images $ARG1 --workers 16 --batch_size 16 & wait

python3 /project/predict/submit.py --submit_path $ARG2
