import argparse
import glob
import os

import torch
import yaml

import experiments
import submodules.haptic_transformer.utils as utils_haptr
from data import helpers
from experiments.clustering_dl_latent import train_fc_autoencoder, train_time_series_autoencoder

torch.manual_seed(42)


def main(args):
    config_file = os.path.join(os.getcwd(), 'config', f"{args.dataset}.yaml")
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if args.overwrite_num_clusters > 0:
        config['num_clusters'] = args.overwrite_num_clusters
        print(f"Overwrited default num clusters. Current number: {args.overwrite_num_clusters}.")
    print(f"Loaded: {config_file}")

    # Dataset
    total_dataset = helpers.load_dataset(config)
    base_dir = os.path.join(os.getcwd(), 'results', args.dataset)
    print(f"Results: {base_dir}")

    # Run a specified experiment
    if args.experiment == "analyze":
        # assumes that results are under "base_dir/{dataset_name}/**/{method_name}.pickle" in the following dict format:
        # {
        #   "x_tsne": [N x 2]
        #   "centroids_tnse": [num_centroids x 2]   (optional for learning-based methods)
        #   "y_supervised": [N]
        #   "y_unsupervised": [N],
        #   "metrics": [K]
        # }
        experiments.analyze_clustering_results(base_dir)

    elif args.experiment == "silhouette":

        if args.ae_load_path == "":
            embeddings = total_dataset.signals

        else:
            autoencoder = torch.load(args.ae_load_path).cpu().eval()
            embeddings = autoencoder.encoder(torch.Tensor(total_dataset.signals)).detach().numpy()

        if args.dec_load_folder == "":
            dec_models = None
        else:
            models_paths = os.path.join(args.dec_load_folder, "**", "best_clustering_model.pt")
            dec_models = sorted(glob.glob(models_paths, recursive=True))

        experiments.silhouette(embeddings, dec_models=dec_models)

    else:
        log_dir = utils_haptr.log.logdir_name(base_dir, args.experiment)
        utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

        # Sci-kit Learn ML algorithms
        if args.experiment == "ml_raw":
            experiments.clustering_ml_raw(total_dataset, log_dir, config['num_clusters'])

        # DEC clustering of raw signals (centroids with the same size as input signals)
        elif args.experiment == "dl_raw":
            experiments.clustering_dl(total_dataset, log_dir, args, config['num_clusters'])

        # DEC clustering of latent vecotrs (trains autoencoder before the clustering)
        elif args.experiment == "dl_latent":

            # train & save or load the autoencoder
            if args.ae_load_path == "":
                if len(total_dataset.signals.shape) == 2:
                    autoencoder = train_fc_autoencoder(total_dataset, log_dir, args)
                else:
                    autoencoder = train_time_series_autoencoder(total_dataset, log_dir, args)
            else:
                autoencoder = torch.load(args.ae_load_path)

            experiments.clustering_dl(total_dataset, log_dir, args, config['num_clusters'], autoencoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--dataset', type=str, default="",
                        choices=['biotac2', 'put', 'touching'])
    parser.add_argument('--experiment', type=str, default="",
                        choices=['ml_raw', 'dl_raw', 'dl_latent', 'analyze', 'silhouette'])

    # deep learning (common for all types of experiments)
    parser.add_argument('--overwrite-num-clusters', type=int, default=-1)
    parser.add_argument('--epochs-ae', type=int, default=500)
    parser.add_argument('--epochs-dec', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batch-size-ae', type=int, default=256)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--latent-size', type=int, default=25, help="Works only for 1D data.")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-5)
    parser.add_argument('--ae-load-path', type=str, default="")
    parser.add_argument('--dec-load-folder', type=str, default="")

    args, _ = parser.parse_known_args()
    main(args)
