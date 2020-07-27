#!/bin/bash
# launch tensorboard from spacenet6_dev container

# set log_dir
LOG_ROOT=/logs
if [ $# -eq 1 ]; then
    LOG_ROOT=$1
fi

# launch tensorboard from project root directory
tensorboard --logdir ${LOG_ROOT} #--bind_all
