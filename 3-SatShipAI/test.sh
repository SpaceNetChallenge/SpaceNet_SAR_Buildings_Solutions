#!/bin/bash

testdatapath=$1
outputcsvpath=$2

#wget http://95.216.245.24:8080/Apps/SpaceNet6/trained_models.zip -O trained_models.zip
#unzip -o -d data/ trained_models.zip
python3 ./test.py --testdata $testdatapath --outputfile $outputcsvpath
