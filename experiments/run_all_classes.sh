#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic-unsupervised

nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_0.yaml" >${DATASET}_0.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_1.yaml" >${DATASET}_1.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_2.yaml" >${DATASET}_2.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_3.yaml" >${DATASET}_3.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_4.yaml" >${DATASET}_4.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_5.yaml" >${DATASET}_5.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_6.yaml" >${DATASET}_6.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/${DATASET}_7.yaml" >${DATASET}_7.log &
