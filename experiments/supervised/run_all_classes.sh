#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic-unsupervised

nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_0.yaml" >${DATASET}_0.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_1.yaml" >${DATASET}_1.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_2.yaml" >${DATASET}_2.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_3.yaml" >${DATASET}_3.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_4.yaml" >${DATASET}_4.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_5.yaml" >${DATASET}_5.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_6.yaml" >${DATASET}_6.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_7.yaml" >${DATASET}_7.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_8.yaml" >${DATASET}_8.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_9.yaml" >${DATASET}_9.log &&
  nohup python -u train_supervised.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type haptr --dataset-config-file "/home/mbed/Projects/haptic-unsupervised/config/supervised/${DATASET}_10.yaml" >${DATASET}_10.log &
