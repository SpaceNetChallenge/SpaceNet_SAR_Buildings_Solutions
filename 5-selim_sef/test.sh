#!/usr/bin/env bash

DATA=$1
OUT_FILE=$2
OUT_DIR=/wdata/results/multiscale
mkdir -p $OUT_DIR

python predict_test.py --gpu 0 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run1_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run1 &
python predict_test.py --gpu 1 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run2_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run2 &
python predict_test.py --gpu 2 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run3_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run3 &
python predict_test.py --gpu 3 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run4_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run4 &
wait
python predict_test.py --gpu 0 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run5_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run5 &
python predict_test.py --gpu 1 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run6_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run6 &
python predict_test.py --gpu 2 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run7_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run7 &
python predict_test.py --gpu 3 --data-path $DATA --dir $OUT_DIR --config configs/b5.json --model weights/run8_eff_unet_tf_efficientnet_b5_ns_0_last --prefix b5run8 &
wait
python predict_test.py --gpu 0 --data-path $DATA --dir $OUT_DIR --config configs/d92.json --model weights/run1_dpn_unet_dpn92_0_last --prefix d92run1 &
python predict_test.py --gpu 1 --data-path $DATA --dir $OUT_DIR --config configs/d92.json --model weights/run2_dpn_unet_dpn92_0_last --prefix d92run2 &
python predict_test.py --gpu 2 --data-path $DATA --dir $OUT_DIR --config configs/d92.json --model weights/run3_dpn_unet_dpn92_0_last --prefix d92run3 &
python predict_test.py --gpu 3 --data-path $DATA --dir $OUT_DIR --config configs/d92.json --model weights/run4_dpn_unet_dpn92_0_last --prefix d92run4 &
wait
python predict_test.py --gpu 0 --data-path $DATA --dir $OUT_DIR --config configs/rx101.json --model weights/run1_resnext_unet_resnext101_0_last --prefix r101run1 &
python predict_test.py --gpu 1 --data-path $DATA --dir $OUT_DIR --config configs/rx101.json --model weights/run2_resnext_unet_resnext101_0_last --prefix r101run2 &
python predict_test.py --gpu 2 --data-path $DATA --dir $OUT_DIR --config configs/rx101.json --model weights/run3_resnext_unet_resnext101_0_last --prefix r101run3 &
python predict_test.py --gpu 3 --data-path $DATA --dir $OUT_DIR --config configs/rx101.json --model weights/run4_resnext_unet_resnext101_0_last --prefix r101run4 &
wait
python ensemble.py --ensembling_cpu_threads 16 --ensembling_dir /wdata/results/ensemble --folds_dir $OUT_DIR

python generate_submission.py --masks-path /wdata/results/ensemble --output-path /wdata/results/solution.csv
cp /wdata/results/solution.csv $OUT_FILE