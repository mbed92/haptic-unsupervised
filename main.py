import argparse
import os

import torch
import yaml
from sklearn.preprocessing import StandardScaler

import experiments
import submodules.haptic_transformer.utils as utils_haptr
from data import helpers

torch.manual_seed(42)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', args.experiment_name)
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load the dataset
    train_ds, val_ds, test_ds = helpers.load_dataset(config)
    total_dataset = train_ds + test_ds

    if len(total_dataset.signals[0].shape) == 1:
        total_dataset.signals = StandardScaler().fit_transform(total_dataset.signals)

    # run a specified experiment
    if args.experiment_name == "clustering_ml_raw":
        experiments.clustering_ml_raw(total_dataset, log_dir)

    elif args.experiment_name == "clustering_dl_raw":
        experiments.clustering_dl_raw(total_dataset, log_dir, args)

    elif args.experiment_name == "clustering_dl_latent":
        experiments.clustering_dl_latent(total_dataset, log_dir, args)

    # elif args.experiment_name == "clustering_dl_latent_with_attention":
    #     experiments.clustering.clustering_dl_latent_with_attention(train_ds, test_ds, log_dir)

    else:
        print("Unknown experiment name. Available choices are: "
              "clustering_ml_raw, clustering_dl_raw, clustering_dl_latent, clustering_dl_latent_with_attention")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str, default="./config/touching.yaml")
    parser.add_argument('--experiment-name', type=str, default="clustering_dl_latent")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str, default="")
    args, _ = parser.parse_known_args()
    main(args)
