import copy
import os
import os.path
import pickle
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from numpy import inf
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models import dec, autoencoders
from utils.sklearn_benchmark import RANDOM_SEED, SCIKIT_LEARN_PARAMS

sns.set()


def _get_embeddings_dataloader(autoencoder, dataloader, device):
    tmp = copy.deepcopy(dataloader)
    with torch.no_grad():
        autoencoder = autoencoder.cpu()
        x_train = torch.Tensor(dataloader.dataset.signals).cpu()
        tmp.dataset.signals = autoencoder.encoder(x_train).numpy()
    autoencoder.to(device)
    return tmp


def clustering_dl(total_dataset: Dataset, log_dir: str, args: Namespace, expected_num_clusters: int,
                  autoencoder: nn.Module = None):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # clustering model requires flattened data
    if len(total_dataset.signals.shape) > 2:
        total_dataset.signals = np.reshape(total_dataset.signals, newshape=(total_dataset.signals.shape[0], -1))

    # create the data loader
    original_dataloader = DataLoader(total_dataset, batch_size=args.batch_size, shuffle=True)
    if autoencoder is not None:
        clustering_dataloader = _get_embeddings_dataloader(autoencoder, original_dataloader, device)
    else:
        clustering_dataloader = copy.deepcopy(original_dataloader)

    inputs = clustering_dataloader.dataset.signals
    if len(inputs.shape) > 2:
        data_shape = inputs.shape[-2] * inputs.shape[-1]
    else:
        data_shape = inputs.shape[-1]

    clust_model = dec.ClusteringModel(expected_num_clusters, data_shape)
    clust_model.centroids = clust_model.set_kmeans_centroids(inputs, expected_num_clusters)
    clust_model.to(device)
    summary(clust_model, input_size=data_shape)

    # setup optimizer
    backprop_config = autoencoders.ops.BackpropConfig()
    backprop_config.optimizer = torch.optim.AdamW
    backprop_config.model = clust_model
    backprop_config.lr = args.lr
    backprop_config.eta_min = args.eta_min
    backprop_config.epochs = args.epochs_dec
    backprop_config.weight_decay = args.weight_decay
    optimizer, scheduler = autoencoders.ops.backprop_init(backprop_config)

    # train the clust_model
    with SummaryWriter(log_dir=log_dir) as writer:
        best_loss = inf
        best_epoch = 0
        best_model = None
        best_metrics = None

        # train epoch
        for epoch in range(args.epochs_dec):

            # clustering assignment
            loss, metrics = dec.ops.train_epoch(clust_model, clustering_dataloader, optimizer, device)

            # go to next epoch if training collapsed (e.g. all clusters are assigned to one cluster)
            if None in [loss, metrics]:
                continue

            # gather metrics
            writer.add_scalar(f"CLUSTERING/train/{loss.name}", loss.get(), epoch)
            for metric in metrics:
                writer.add_scalar(f"CLUSTERING/train/{metric.name}", metric.get(), epoch)
            writer.add_scalar(f'CLUSTERING/train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            scheduler.step()

            # save the best model and log metrics
            current_loss = loss.get()
            if current_loss < best_loss:
                torch.save(clust_model, os.path.join(writer.log_dir, 'best_clustering_model'))
                best_model = copy.deepcopy(clust_model)
                best_loss = current_loss
                best_epoch = epoch
                best_metrics = [(m.name, m.get()) for m in metrics]
                [print(f"{m.name} {m.get()}") for m in metrics]

            print(f"Epoch: {epoch} / {args.epochs_dec}. Best loss: {best_loss} for epoch {best_epoch}")

        scheduler.step()

    # verify the last model
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()

            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=SCIKIT_LEARN_PARAMS["figsize"])
            log_picture = os.path.join(log_dir, "tsne.png")

            x, y = clustering_dataloader.dataset.signals, clustering_dataloader.dataset.labels
            x = np.reshape(x, newshape=(x.shape[0], -1))

            outputs = best_model(torch.Tensor(x))
            y_pred = torch.argmax(outputs, -1).detach().cpu().numpy()

            # draw tsne for the best model with learned centroids
            centroids = best_model.centroids.numpy()

            # setup colors
            colors = plt.cm.rainbow(np.linspace(0, 1, expected_num_clusters))

            # plot TSNE
            tsne = TSNE(n_components=SCIKIT_LEARN_PARAMS["tsne_n_components"])
            num_points = len(clustering_dataloader.dataset.signals)
            embeddings = np.concatenate([clustering_dataloader.dataset.signals, centroids])
            x_tsne = tsne.fit_transform(embeddings)

            ax.set_title('DEC', size=18)
            ax.scatter(x_tsne[:num_points, 0], x_tsne[:num_points, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)
            ax.scatter(x_tsne[num_points:, 0], x_tsne[num_points:, 1], c='black', s=200)

            # save embeddings
            file_handler = open(os.path.join(log_dir, "DEC.pickle"), "wb")
            pickle.dump({
                "x_tsne": x_tsne[:num_points],
                "centroids_tsne": x_tsne[num_points:],
                "y_supervised": y,
                "y_unsupervised": y_pred,
                "metrics": best_metrics
            }, file_handler)

        # save tsne
        plt.savefig(log_picture, dpi=fig.dpi)
        plt.close(fig)
