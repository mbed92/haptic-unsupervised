import copy
import os
from argparse import Namespace

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from models import autoencoders
from models.autoencoders.conv import TimeSeriesConvAutoencoderConfig, TimeSeriesConvAutoencoder
from models.autoencoders.fc import FullyConnectedAutoencoder, FullyConnectedAutoencoderConfig
from .clustering_dl_raw import clustering_dl_raw
from .benchmark import RANDOM_SEED

sns.set()


def train_time_series_autoencoder(ds: Dataset, log_dir: str, args: Namespace):
    train_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data_shape = train_dl.dataset.signals.shape[-2:]

    # set up a model (find the best config)
    nn_params = TimeSeriesConvAutoencoderConfig()
    nn_params.data_shape = data_shape
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    nn_params.dropout = args.dropout
    autoencoder = TimeSeriesConvAutoencoder(nn_params)
    device = autoencoders.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # train the main autoencoder
    backprop_config = autoencoders.ops.BackpropConfig()
    backprop_config.model = autoencoder
    backprop_config.optimizer = torch.optim.AdamW
    backprop_config.lr = args.lr
    backprop_config.eta_min = args.eta_min
    backprop_config.epochs = args.epochs_ae
    backprop_config.weight_decay = args.weight_decay

    return train_autoencoder(train_dl, log_dir, args, backprop_config, autoencoder, device)


def train_fc_autoencoder(total_dataset: Dataset, log_dir: str, args: Namespace):
    train_dl = DataLoader(total_dataset, batch_size=args.batch_size, shuffle=True)
    data_shape = train_dl.dataset.signals.shape[-1]

    # set up a model (find the best config)
    nn_params = FullyConnectedAutoencoderConfig()
    nn_params.data_shape = [data_shape]
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    nn_params.dropout = args.dropout
    nn_params.latent_size = args.latent_size
    autoencoder = FullyConnectedAutoencoder(nn_params)
    device = autoencoders.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # train the main autoencoder
    backprop_config = autoencoders.ops.BackpropConfig()
    backprop_config.model = autoencoder
    backprop_config.optimizer = torch.optim.AdamW
    backprop_config.lr = args.lr
    backprop_config.eta_min = args.eta_min
    backprop_config.epochs = args.epochs_ae
    backprop_config.weight_decay = args.weight_decay

    return train_autoencoder(train_dl, log_dir, args, backprop_config, autoencoder, device)


def train_autoencoder(total_dataset: DataLoader, log_dir, args, backprop_config, autoencoder, device):
    best_loss = 9999.9
    best_model = None
    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = autoencoders.ops.backprop_init(backprop_config)

        for epoch in range(args.epochs_ae):
            train_loss, exemplary_sample = autoencoders.ops.train_epoch(autoencoder, total_dataset, optimizer, device)
            writer.add_scalar(f'AE/train/{train_loss.name}', train_loss.get(), epoch)
            writer.add_scalar(f'AE/train/lr', optimizer.param_groups[0]['lr'], epoch)
            if exemplary_sample is not None:
                writer.add_image('AE/train/image', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()
            scheduler.step()

            # save the best autoencoder
            current_test_loss = train_loss.get()
            if current_test_loss < best_loss:
                torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(autoencoder)

    return best_model


def clustering_dl_latent(total_dataset: Dataset, log_dir: str, args: Namespace, expected_num_clusters: int):
    torch.manual_seed(RANDOM_SEED)

    # prepare the autoencoder (should work on data with shapes NxC or NxCxL)
    shape = total_dataset.signals.shape
    create_fc_autoencoder = False
    if len(shape) == 2:
        create_fc_autoencoder = True

    # train & save or load the autoencoder
    if args.load_path == "":
        if create_fc_autoencoder:
            autoencoder = train_fc_autoencoder(total_dataset, log_dir, args)
        else:
            autoencoder = train_time_series_autoencoder(total_dataset, log_dir, args)

        torch.save(autoencoder, os.path.join(log_dir, 'test_model'))
    else:
        autoencoder = torch.load(args.load_path)

    # prepare a new dataset with latent representations
    with torch.no_grad():
        autoencoder = autoencoder.cpu()
        x_train = torch.Tensor(total_dataset.signals).cpu()
        total_dataset.signals = autoencoder.encoder(x_train).numpy()

    clustering_dl_raw(total_dataset, log_dir, args, expected_num_clusters)
