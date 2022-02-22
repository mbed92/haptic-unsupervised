#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS_SAE=$3
EPOCHS_AE=$4

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic-unsupervised

nohup python -u train_ae.py --epochs-sae ${EPOCHS_SAE} --epochs-ae ${EPOCHS_AE} --batch-size ${BATCH} --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/unsupervised/${DATASET}.yaml" >${DATASET}.log &
#  nohup python -u train_ae.py --epochs-sae ${EPOCHS_SAE} --epochs-ae ${EPOCHS_AE} --batch-size ${BATCH} --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/unsupervised/touching.yaml" >touching.log &
