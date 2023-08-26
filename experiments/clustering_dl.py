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
from utils.clustering import distribution_hardening
from utils.sklearn_benchmark import RANDOM_SEED, SCIKIT_LEARN_PARAMS

sns.set()


def _get_embeddings(autoencoder, dataset, device):
    autoencoder.train(False)
    with torch.no_grad():
        autoencoder = autoencoder.cpu()
        x = torch.Tensor(dataset).cpu()
        x = autoencoder.encoder(x)
        autoencoder.to(device)
    return x.numpy()


def _create_data_loader(autoencoder, dataset, device, args):
    if autoencoder is None:
        signals_dataloader = None
        dataset.signals = np.reshape(dataset.signals, newshape=(dataset.signals.shape[0], -1))
        clustering_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        signals_dataloader = DataLoader(dataset, batch_size=args.batch_size_ae, shuffle=True)
        clustering_dataset = copy.deepcopy(dataset)
        clustering_dataset.signals = _get_embeddings(autoencoder, clustering_dataset.signals, device)
        clustering_dataloader = DataLoader(clustering_dataset, batch_size=args.batch_size, shuffle=True)

    return signals_dataloader, clustering_dataloader


def clustering_dl(total_dataset: Dataset,
                  log_dir: str,
                  args: Namespace,
                  expected_num_clusters: int,
                  autoencoder: nn.Module = None):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create the data loader
    signals_dataloader, clustering_dataloader = _create_data_loader(autoencoder, total_dataset, device, args)

    # setup dec for training
    inputs = clustering_dataloader.dataset.signals
    clust_model = dec.ClusteringModel(expected_num_clusters, inputs.shape[-1])
    clust_model.centroids = clust_model.set_kmeans_centroids(inputs, expected_num_clusters)
    clust_model.to(device)
    main_optimizer, main_scheduler = clust_model.setup(args)
    summary(clust_model, input_size=inputs.shape[-1])

    # setup autoencoder for fine-tuning
    ae_optimizer, ae_scheduler = None, None
    if autoencoder is not None:
        if args.lr is not None:
            args.lr /= 10.0  # reduce learning rate 10x for fine-tuning autoencoder
        ae_optimizer, ae_scheduler = autoencoder.setup(args)

    # train the clust_model
    with SummaryWriter(log_dir=log_dir) as writer:
        best_metric = inf
        best_epoch = 0
        best_model = None
        best_metrics = None

        # train epoch
        for epoch in range(args.epochs_dec):

            # update the target distribution
            if epoch % 100 == 0:
                clust_model.cpu()
                clust_model.train(False)
                soft_assignments = clust_model.t_student(torch.Tensor(inputs)).detach()
                target_distribution = distribution_hardening(soft_assignments)
                clustering_dataloader.dataset.p_target = target_distribution.numpy().astype(np.float64)
                clust_model.train(True)
                clust_model.to(device)

            # clustering assignment
            loss, metrics = dec.ops.train_epoch(clust_model, clustering_dataloader, main_optimizer, device)

            # fine-tune autoencoder each 10 epoch
            if None not in [autoencoder, ae_scheduler, ae_optimizer, signals_dataloader] and epoch % 5 == 0:
                ae_loss, _ = autoencoders.ops.train_epoch(autoencoder, signals_dataloader, ae_optimizer, device)
                writer.add_scalar(f"RECONSTRUCTION/train/{ae_loss.name}", ae_loss.get(), epoch)
                ae_scheduler.step()
                clustering_dataloader.dataset.signals = _get_embeddings(autoencoder,
                                                                        signals_dataloader.dataset.signals, device)

            # go to next epoch if training collapsed (e.g. all clusters are assigned to one cluster)
            if None in [loss, metrics]:
                continue

            # gather metrics
            writer.add_scalar(f"CLUSTERING/train/{loss.name}", loss.get(), epoch)
            for metric in metrics:
                writer.add_scalar(f"CLUSTERING/train/{metric.name}", metric.get(), epoch)
            writer.add_scalar(f'CLUSTERING/train/lr', main_optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            main_scheduler.step()

            # save the best model and log metrics
            performance_metric = None
            for m in metrics:
                if "accuracy" in m.name.lower():
                    performance_metric = -m.get()
            if performance_metric is None:
                performance_metric = loss.get()

            if performance_metric < best_metric:
                torch.save(clust_model, os.path.join(writer.log_dir, 'best_clustering_model.pt'))
                best_model = copy.deepcopy(clust_model)
                best_metric = performance_metric
                best_epoch = epoch
                best_metrics = [(m.name, m.get()) for m in metrics]
                [print(f"{m.name} {m.get()}") for m in metrics]

                if autoencoder is not None:
                    torch.save(autoencoder, os.path.join(writer.log_dir, 'best_autoencoder.pt'))

            print(f"Epoch: {epoch} / {args.epochs_dec}. Best metric: {best_metric} for epoch {best_epoch}")

    # verify the last model
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()

            fig, ax = plt.subplots(constrained_layout=True, figsize=SCIKIT_LEARN_PARAMS["figsize"])
            x, y = clustering_dataloader.dataset.signals, clustering_dataloader.dataset.labels
            x = np.reshape(x, newshape=(x.shape[0], -1))
            outputs = best_model(torch.Tensor(x))
            y_pred = torch.argmax(outputs, -1).detach().cpu().numpy()

            # plot TSNE
            colors = plt.cm.rainbow(np.linspace(0, 1, expected_num_clusters))
            centroids = best_model.centroids.numpy()
            tsne = TSNE(n_components=SCIKIT_LEARN_PARAMS["tsne_n_components"])
            num_points = len(clustering_dataloader.dataset.signals)
            embeddings = np.concatenate([clustering_dataloader.dataset.signals, centroids])
            x_tsne = tsne.fit_transform(embeddings)

            # save embeddings
            file_handler = open(os.path.join(log_dir, "DEC.pickle"), "wb")
            pickle.dump({
                "x_tsne": x_tsne[:num_points],
                "centroids_tsne": x_tsne[num_points:],
                "centroids_raw": centroids,
                "y_supervised": y,
                "y_unsupervised": y_pred,
                "metrics": best_metrics,
                "embeddings": embeddings[:num_points]
            }, file_handler)

            # visualize
            ax.set_title('DEC', size=18)
            ax.scatter(x_tsne[:num_points, 0], x_tsne[:num_points, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)
            ax.scatter(x_tsne[num_points:, 0], x_tsne[num_points:, 1], c='black', s=200)

        # save tsne
        log_picture = os.path.join(log_dir, "tsne.png")
        plt.savefig(log_picture, dpi=fig.dpi)
        plt.close(fig)
