import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

import experiments
import submodules.haptic_transformer.utils as utils_haptr
import utils.sklearn_benchmark
from data import helpers

torch.manual_seed(42)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', args.experiment_name)
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load the dataset
    train_ds, _, test_ds = helpers.load_dataset(config)
    total_dataset = train_ds + test_ds
    expected_num_clusters = total_dataset.num_classes

    if len(total_dataset.signals[0].shape) == 1:
        total_dataset.signals = StandardScaler().fit_transform(total_dataset.signals)

    # run a specified experiment
    if args.experiment_name == "clustering_ml_raw":
        experiments.clustering_ml_raw(total_dataset, log_dir, expected_num_clusters)

    elif args.experiment_name == "clustering_dl_raw":
        experiments.clustering_dl_raw(total_dataset, log_dir, args, expected_num_clusters)

    elif args.experiment_name == "clustering_dl_latent":
        experiments.clustering_dl_latent(total_dataset, log_dir, args, expected_num_clusters)

    # assumes that results are under ./args.results_folder/{method_name}.pickle in the following dict format:
    # {
    #   "x_tsne": [N x 2]
    #   "centroids_tnse": [num_centroids x 2]   (optional for learning-based methods)
    #   "y_supervised": [N]
    #   "y_unsupervised": [N],
    #   "metrics": [K]
    # }
    elif args.experiment_name == "analyze_clustering_results":
        experiments.analyze_clustering_results(total_dataset, args.results_folder)

    else:
        print("Unknown experiment name. Available choices are: "
              "clustering_ml_raw, clustering_dl_raw, clustering_dl_latent, analyze_clustering_results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--dataset-config-file', type=str, default="./config/biotac2.yaml")
    parser.add_argument('--experiment-name', type=str, default="clustering_ml_raw")

    # deep learning
    parser.add_argument('--epochs-ae', type=int, default=200)
    parser.add_argument('--epochs-dec', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--latent-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str, default="")

    # analysis
    parser.add_argument('--results-folder', type=str, default="./experiments/results")

    args, _ = parser.parse_known_args()
    main(args)
