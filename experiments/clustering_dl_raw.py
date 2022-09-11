import copy
import os
import os.path
import pickle
from argparse import Namespace
from itertools import islice, cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import utils
import utils.sklearn_benchmark
from data import helpers
from models import dec
from .benchmark import DEFAULT_PARAMS, RANDOM_SEED, COLOR_BASE

sns.set()


def clustering_dl_raw(train_ds: Dataset, test_ds: Dataset, log_dir: str, args: Namespace):
    torch.manual_seed(RANDOM_SEED)

    # clustering model requires flattened data
    total_dataset = train_ds + test_ds
    if len(total_dataset.signals.shape) > 2:
        total_dataset.signals = np.reshape(total_dataset.signals, newshape=(total_dataset.signals.shape[0], -1))

    # get all the data for further calculations (remember that you always need to iterate over some bigger datasets)
    dataloader = DataLoader(total_dataset, batch_size=args.batch_size, shuffle=True)
    x_train, y_train = helpers.get_total_data_from_dataloader(dataloader)

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    shape = train_ds.signals.shape
    if len(shape) > 2:
        data_shape = shape[-2] * shape[-1]
    else:
        data_shape = shape[-1]

    clust_model = dec.ClusteringModel(DEFAULT_PARAMS["n_clusters"], data_shape)
    summary(clust_model, input_size=data_shape)

    # setup optimizer
    backprop_config = utils.ops.BackpropConfig()
    backprop_config.optimizer = torch.optim.AdamW
    backprop_config.model = clust_model
    backprop_config.lr = args.lr
    backprop_config.eta_min = args.eta_min
    backprop_config.epochs = args.epochs_per_run * args.runs
    backprop_config.weight_decay = args.weight_decay
    optimizer, scheduler = utils.ops.backprop_init(backprop_config)

    # train the clust_model model
    best_loss = 9999.9
    best_model = None
    with SummaryWriter(log_dir=log_dir) as writer:
        for run in range(args.runs):

            # create a ClusteringDataset that consists of x, y, and soft assignments
            train_dataloader = dec.ops.add_soft_predictions(clust_model, x_train, y_train, device, args.batch_size)

            for epoch in range(args.epochs_per_run):
                step = run * args.epochs_per_run + epoch

                # train epoch
                loss, metrics = dec.ops.train_epoch(clust_model, train_dataloader, optimizer, device)
                writer.add_scalar(f"CLUSTERING/train/{loss.name}", loss.get(), step)
                for metric in metrics:
                    writer.add_scalar(f"CLUSTERING/train/{metric.name}", metric.get(), step)
                writer.add_scalar(f'CLUSTERING/train/lr', optimizer.param_groups[0]['lr'], step)
                writer.flush()
                print(f"Run {run}/{args.runs}. Epoch: {epoch} / {args.epochs_per_run}. Best loss: {loss.get()}")

            scheduler.step()

            # save the best model
            current_test_loss = loss.get()
            if current_test_loss < best_loss:
                torch.save(clust_model, os.path.join(writer.log_dir, 'best_model'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(clust_model)

    # verify the last model
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()

            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(15, 15))
            log_picture = os.path.join(log_dir, "tsne.png")

            x, y = total_dataset.signals, total_dataset.labels
            x = np.reshape(x, newshape=(x.shape[0], -1))
            outputs = best_model(torch.Tensor(x))
            y_pred = torch.argmax(outputs, -1).detach().cpu().numpy()

            # draw tsne for the best model with learned centroids
            centroids = best_model.centroids.numpy()

            # setup colors
            colors = np.array(list(islice(cycle(COLOR_BASE), DEFAULT_PARAMS["n_clusters"], )))
            colors = np.append(colors, ["#000000"])

            # plot TSNE
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(train_ds.signals)
            y_tsne = tsne.fit_transform(centroids)
            ax.set_title('DEC', size=18)
            ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)
            ax.scatter(y_tsne[:, 0], y_tsne[:, 1], c='black', s=200)

            # save embeddings
            file_handler = open(os.path.join(log_dir, "dec.pickle"), "wb")
            pickle.dump({
                "tsne": x_tsne,
                "predicted_labels": y_pred
            }, file_handler)

        # save tsne
        plt.savefig(log_picture, dpi=fig.dpi)
        plt.show()
        plt.close(fig)
