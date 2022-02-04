#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic-unsupervised

nohup python -u train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}.yaml" >${DATASET}_1.log &
