import copy
import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from models import autoencoders
from models.autoencoders.conv import TimeSeriesConvAutoencoderConfig, TimeSeriesConvAutoencoder
from models.autoencoders.fc import FullyConnectedAutoencoder, FullyConnectedAutoencoderConfig
from utils.sklearn_benchmark import RANDOM_SEED


def _train_autoencoder(total_dataset: DataLoader, log_dir, args, autoencoder, device):
    torch.manual_seed(RANDOM_SEED)
    best_loss = 9999.9
    best_model = None

    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = autoencoder.setup(args)

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
                torch.save(autoencoder, os.path.join(writer.log_dir, 'autoencoder.pt'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(autoencoder)

            print(f"Autoencoder training. Epoch: {epoch}, best loss: {best_loss}")

    return best_model


def train_time_series_autoencoder(ds: Dataset, log_dir: str, args: Namespace):
    train_dl = DataLoader(ds, batch_size=args.batch_size_ae, shuffle=True)
    data_shape = train_dl.dataset.signals.shape[-2:]

    nn_params = TimeSeriesConvAutoencoderConfig()
    nn_params.data_shape = data_shape
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    autoencoder = TimeSeriesConvAutoencoder(nn_params)
    device = autoencoders.ops.hardware_upload(autoencoder, nn_params.data_shape)

    return _train_autoencoder(train_dl, log_dir, args, autoencoder, device)


def train_fc_autoencoder(total_dataset: Dataset, log_dir: str, args: Namespace):
    train_dl = DataLoader(total_dataset, batch_size=args.batch_size_ae, shuffle=True)
    data_shape = train_dl.dataset.signals.shape[-1]

    nn_params = FullyConnectedAutoencoderConfig()
    nn_params.data_shape = [data_shape]
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    nn_params.latent_size = args.latent_size
    autoencoder = FullyConnectedAutoencoder(nn_params)
    device = autoencoders.ops.hardware_upload(autoencoder, nn_params.data_shape)

    return _train_autoencoder(train_dl, log_dir, args, autoencoder, device)
