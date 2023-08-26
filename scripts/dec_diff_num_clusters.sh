#!/bin/bash

ROOT=$(readlink -f "${PWD}/..")
cd ${ROOT}

DATASET=$1
EXP=$2

# Run experiments to find the optimal number of clusters.
nohup python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 2 &&
  python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 3 &&
  python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 4 &&
  python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 5 &&
  python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 6 &&
  python -u main.py --dataset ${DATASET} --experiment ${EXP} --overwrite-num-clusters 7 &
