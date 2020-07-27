#!/bin/bash

# set image name
IMAGE="spacenet6:dev"

# get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# build docker image from project root directory
cd ${PROJ_DIR} && \
docker build -t ${IMAGE} -f ${THIS_DIR}/Dockerfile.dev .
