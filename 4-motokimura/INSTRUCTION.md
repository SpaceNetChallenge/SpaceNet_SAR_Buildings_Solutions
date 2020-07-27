# Instructions for Final Scoring

This document provides instructions for the final testing/scoring of motokimura's solution.

## Prepare SpaceNet6 data

```
DATA_DIR=${HOME}/data  # path to download SpaceNet6 dataset
mkdir -p ${DATA_DIR}

# download and extract train data
cd ${DATA_DIR}
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_train.tar.gz

# download and extract test data
cd ${DATA_DIR}
aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz .
tar -xvf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz

# please prepare private test data under `DATA_DIR`!
```

## Prepare temporal directory

```
WDATA_DIR=${HOME}/wdata  # path to directory for temporal files (train logs, prediction results, etc.)
mkdir -p ${WDATA_DIR}
```

## Build image

```
cd ${CODE_DIR}  # `code` directory containing `Dockerfile`, `train.sh`, `test.sh`, and etc. 
nvidia-docker build -t motokimura .
```

During the build, my home built models are downloaded
so that `test.sh` can run without re-training the models.

## Prepare container

```
# launch container
nvidia-docker run --ipc=host -v ${DATA_DIR}:/data:ro -v ${WDATA_DIR}:/wdata -it motokimura
```

It's necessary to add `--ipc=host` option when run docker (as written in [flags.txt](flags.txt)).
Otherwise multi-threaded PyTorch dataloader will crash.

## Train

**WARNINGS: `train.sh` updates my home built models downloaded during docker build.**

```
# start training!
(in container) ./train.sh /data/train/AOI_11_Rotterdam

# if you need logs:
(in container) ./train.sh /data/train/AOI_11_Rotterdam 2>&1 | tee /wdata/train.log
```

Note that this is a sample call of `train.sh`. 
i.e., you need to specify the correct path to training data folder.

## Test

```
# start testing!
(in container) ./test.sh /data/test_public/AOI_11_Rotterdam /wdata/solution.csv

# if you need logs:
(in container) ./test.sh /data/test_public/AOI_11_Rotterdam /wdata/solution.csv 2>&1 | tee /wdata/test.log
```

Note that this is a sample call of `test.sh`. 
i.e., you need to specify the correct paths to testing image folder and output csv file.
