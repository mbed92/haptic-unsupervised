#!/usr/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/mbed/Projects/haptic-unsupervised"

SCRIPT="/home/mbed/Projects/haptic-unsupervised/experiments/train_dec.py"

nohup python3 -u $SCRIPT --num-clusters 2 >log_2.txt
  python3 -u $SCRIPT --num-clusters 3 >log_3.txt &&
  python3 -u $SCRIPT --num-clusters 4 >log_4.txt &&
  python3 -u $SCRIPT --num-clusters 5 >log_5.txt &&
  python3 -u $SCRIPT --num-clusters 6 >log_6.txt &&
  python3 -u $SCRIPT --num-clusters 7 >log_7.txt &&
  python3 -u $SCRIPT --num-clusters 8 >log_8.txt &&
  python3 -u $SCRIPT --num-clusters 9 >log_9.txt &
