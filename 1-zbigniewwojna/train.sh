#nohup python split_folds.py "$@" > /wdata/logs/create_masks.out &
#wait
#traindataargs="--sardir $traindatapath/SAR-Intensity --opticaldir $traindatapath/PS-RGB --labeldir $traindatapath/geojson_buildings --rotationfile $traindatapath/SummaryData/SAR_orientations.txt "
#./baseline.py --pretrain --train $traindataargs 

source activate py36
traindatapath=$1
timeout=22h
arg2='--dec_ch 32 64 128 256 256'
mkdir -p /wdata/logs

python main.py --split_folds --train_data_folder $traindatapath

CUDA_VISIBLE_DEVICES=0 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 0 | tee /wdata/logs/fold_0.out &
sleep 60 #to download pretrained weights
CUDA_VISIBLE_DEVICES=1 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 3 | tee /wdata/logs/fold_3.out &
CUDA_VISIBLE_DEVICES=2 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 6 | tee /wdata/logs/fold_6.out &
CUDA_VISIBLE_DEVICES=3 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 9 | tee /wdata/logs/fold_9.out &
wait
pkill -f train_data_folder

CUDA_VISIBLE_DEVICES=0 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 1 | tee /wdata/logs/fold_1.out &
CUDA_VISIBLE_DEVICES=1 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 2 | tee /wdata/logs/fold_2.out &
CUDA_VISIBLE_DEVICES=2 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 7 | tee /wdata/logs/fold_7.out &
CUDA_VISIBLE_DEVICES=3 nohup timeout $timeout python main.py --train_data_folder $traindatapath --train --val --fold 8 | tee /wdata/logs/fold_8.out &
wait
pkill -f train_data_folder

