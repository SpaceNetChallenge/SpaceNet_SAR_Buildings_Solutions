DATA=$1

echo "Preparing masks..."
python preprocessing/prepare_masks.py --data-path $DATA --labels-dir /wdata/labels --masks-dir /wdata/masks

echo "Training models..."

python train.py --gpu 0 --config configs/b5.json --opt-level O1 --prefix run1_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b51 &
python train.py --gpu 1 --config configs/b5.json --opt-level O1 --prefix run2_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b52 &
python train.py --gpu 2 --config configs/b5.json --opt-level O1 --prefix run3_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b53 &
python train.py --gpu 3 --config configs/b5.json --opt-level O1 --prefix run4_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b54 &

wait
python train.py --gpu 0 --config configs/b5.json --opt-level O1 --prefix run5_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b55 &
python train.py --gpu 1 --config configs/b5.json --opt-level O1 --prefix run6_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b56 &
python train.py --gpu 2 --config configs/b5.json --opt-level O1 --prefix run7_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b57 &
python train.py --gpu 3 --config configs/b5.json --opt-level O1 --prefix run8_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/b58 &

wait
python train.py --gpu 0 --config configs/d92.json --opt-level O1 --prefix run1_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/d921 &
python train.py --gpu 1 --config configs/d92.json --opt-level O1 --prefix run2_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/d922 &
python train.py --gpu 2 --config configs/d92.json --opt-level O1 --prefix run3_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/d923 &
python train.py --gpu 3 --config configs/d92.json --opt-level O1 --prefix run4_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/d924 &

wait
python train.py --gpu 0 --config configs/rx101.json --opt-level O1 --prefix run1_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/r1 &
python train.py --gpu 1 --config configs/rx101.json --opt-level O1 --prefix run2_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/r2 &
python train.py --gpu 2 --config configs/rx101.json --opt-level O1 --prefix run3_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/r3 &
python train.py --gpu 3 --config configs/rx101.json --opt-level O1 --prefix run4_ --test_every 5 --output-dir weights/ --data-dir $DATA > logs/r4 &