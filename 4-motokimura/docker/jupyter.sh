#!/bin/bash
# launch jupyter from spacenet6_dev container

# set jupyter port
JUPYTER_PORT=8889
if [ $# -eq 1 ]; then
    JUPYTER_PORT=$1
fi

echo "access jupyter sever via port: ${JUPYTER_PORT}"

# get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# build docker image from project root directory
cd ${PROJ_DIR} && \
jupyter lab --port ${JUPYTER_PORT} --ip=0.0.0.0 --allow-root --no-browser
