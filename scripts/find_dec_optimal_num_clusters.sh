#!/bin/bash

ROOT=$(readlink -f "${PWD}/..")
cd ${ROOT}

nohup bash dec_diff_num_clusters.sh biotac2 ml_raw &&
  bash dec_diff_num_clusters.sh biotac2 dl_raw &&
  bash dec_diff_num_clusters.sh touching ml_raw &&
  bash dec_diff_num_clusters.sh touching dl_raw &&
  bash dec_diff_num_clusters.sh touching dl_latent &
