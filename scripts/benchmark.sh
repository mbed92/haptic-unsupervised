#!/bin/bash

ROOT=$(readlink -f "${PWD}/..")
cd ${ROOT}

## Benchmark. Run experiments one after the other.
nohup python -u main.py --dataset biotac2 --experiment ml_raw > benchmark_biotac2.out &&
  python -u main.py --dataset biotac2 --experiment dl_raw > benchmark_biotac2.out &&
  python -u main.py --dataset biotac2 --experiment dl_latent --ae-load-path ./models/best_autoencoder_biotac2.pt > benchmark_biotac2.out &&
  python -u main.py --dataset touching --experiment ml_raw > benchmark_touching.out &&
  python -u main.py --dataset touching --experiment dl_raw > benchmark_touching.out &&
  python -u main.py --dataset touching --experiment dl_latent --ae-load-path ./models/best_autoencoder_touching.pt > benchmark_touching.out &
