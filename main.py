import argparse
import os

import torch
import yaml

import experiments
import submodules.haptic_transformer.utils as utils_haptr
from data import helpers

torch.manual_seed(42)


def main(args):
    # load data config
    config_file = os.path.join(os.getcwd(), 'config', f"{args.dataset}.yaml")
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load the dataset
    total_dataset = helpers.load_dataset(config)
    base_dir = os.path.join(os.getcwd(), 'results', args.dataset)

    # run a specified experiment
    if args.experiment == "analyze":
        # assumes that results are under "base_dir/{dataset_name}/**/{method_name}.pickle" in the following dict format:
        # {
        #   "x_tsne": [N x 2]
        #   "centroids_tnse": [num_centroids x 2]   (optional for learning-based methods)
        #   "y_supervised": [N]
        #   "y_unsupervised": [N],
        #   "metrics": [K]
        # }
        experiments.analyze_clustering_results(total_dataset, base_dir)
    else:
        log_dir = utils_haptr.log.logdir_name(base_dir, args.experiment)
        utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

        if args.experiment == "ml_raw":
            experiments.clustering_ml_raw(total_dataset, log_dir, config['num_clusters'])

        elif args.experiment == "dl_raw":
            experiments.clustering_dl(total_dataset, log_dir, args, config['num_clusters'])

        elif args.experiment == "dl_latent":
            experiments.clustering_dl_latent(total_dataset, log_dir, args, config['num_clusters'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--dataset', type=str, default="mock",
                        choices=['biotac2', 'put', 'touching', 'mock'])
    parser.add_argument('--experiment', type=str, default="dl_raw",
                        choices=['ml_raw', 'dl_raw', 'dl_latent', 'analyze'])

    # deep learning (common for all types of experiments)
    parser.add_argument('--epochs-ae', type=int, default=100)
    parser.add_argument('--epochs-dec', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--batch-size-ae', type=int, default=256)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--latent-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--ae-load-path', type=str, default="")

    args, _ = parser.parse_known_args()
    main(args)
