#!/bin/bash

dstdir=/root

settings="\
--rotationfilelocal $dstdir/SAR_orientations.txt \
--maskdir $dstdir/masks \
--sarprocdir $dstdir/sartrain \
--opticalprocdir $dstdir/optical \
--traincsv $dstdir/train.csv \
--validcsv $dstdir/valid.csv \
--opticaltraincsv $dstdir/opticaltrain.csv \
--opticalvalidcsv $dstdir/opticalvalid.csv \
--testcsv $dstdir/test.csv \
--yamlpath $dstdir/sar.yaml \
--opticalyamlpath $dstdir/optical.yaml \
--modeldir $dstdir/weights \
--testprocdir $dstdir/sartest \
--testoutdir $dstdir/inference_continuous \
--testbinarydir $dstdir/inference_binary \
--testvectordir $dstdir/inference_vectors \
--rotate \
--transferoptical \
--mintrainsize 20 \
--mintestsize 80 \
"
