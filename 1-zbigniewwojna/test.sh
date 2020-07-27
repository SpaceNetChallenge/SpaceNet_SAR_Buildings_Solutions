source activate py36
testdatapath=$1
outputpath=$2
timeout=70m
arg2='--dec_ch 32 64 128 256 256'
mkdir -p /wdata/logs

CUDA_VISIBLE_DEVICES=0 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 0 | tee /wdata/logs/test_fold_0.out &
CUDA_VISIBLE_DEVICES=1 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 3 | tee /wdata/logs/test_fold_3.out &
CUDA_VISIBLE_DEVICES=2 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 6 | tee /wdata/logs/test_fold_6.out &
CUDA_VISIBLE_DEVICES=3 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 9 | tee /wdata/logs/test_fold_9.out &
wait
pkill -f test_data_folder

CUDA_VISIBLE_DEVICES=0 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 1 | tee /wdata/logs/test_fold_1.out &
CUDA_VISIBLE_DEVICES=1 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 2 | tee /wdata/logs/test_fold_2.out &
CUDA_VISIBLE_DEVICES=2 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 7 | tee /wdata/logs/test_fold_7.out &
CUDA_VISIBLE_DEVICES=3 nohup timeout $timeout python main.py --test_data_folder $testdatapath --test --fold 8 | tee /wdata/logs/test_fold_8.out &
wait
pkill -f test_data_folder

python main.py --merge --merge_folds 0 1 2 3 6 7 8 9  --solution_file $outputpath 
