#!/bin/bash
traindatapath=$1

echo "Train" $traindatapath



#export CUDA_VISIBLE_DEVICES='0'; python3 train.py --basedir $traindatapath
DIR="data"
if [ -d "$DIR" ]; then
  echo " ${DIR} already exists "
else
  echo "Error: ${DIR} not found. Creating now"
  mkdir ${DIR}
 fi

python3 nasiosprepare.py --basedir $traindatapath

export CUDA_VISIBLE_DEVICES='0'; python3 nasiostrain1.py --basedir $traindatapath > /wdata/nas1.txt &
export CUDA_VISIBLE_DEVICES='1'; python3 nasiostrain2.py --basedir $traindatapath > /wdata/nas2.txt &
export CUDA_VISIBLE_DEVICES='2'; python3 train.py --basedir $traindatapath > /wdata/vog.txt &

#
#traindatapath=$1
#traindataargs="\
#--sardir $traindatapath/SAR-Intensity \
#--opticaldir $traindatapath/PS-RGB \
#--labeldir $traindatapath/geojson_buildings \
#--rotationfile $traindatapath/SummaryData/SAR_orientations.txt \
#"
#
#source settings.sh
#
#./baseline.py --pretrain --train $traindataargs $settings
