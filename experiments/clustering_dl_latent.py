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
from . import clustering_dl_raw
from .benchmark import RANDOM_SEED

sns.set()


def train_autoencoder(train_ds: Dataset, test_ds: Dataset, log_dir: str, args: Namespace):
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    data_shape = test_dl.dataset.signals.shape[-2:]

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
    backprop_config.epochs = args.epochs
    backprop_config.weight_decay = args.weight_decay

    # train
    best_loss = 9999.9
    best_model = None
    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = autoencoders.ops.backprop_init(backprop_config)

        for epoch in range(args.epochs):
            train_loss = autoencoders.ops.train_epoch(autoencoder, train_dl, optimizer, device)
            writer.add_scalar(f'AE/train/{train_loss.name}', train_loss.get(), epoch)
            writer.add_scalar(f'AE/train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            scheduler.step()

            # test epoch
            test_loss, exemplary_sample = autoencoders.ops.test_epoch(autoencoder, test_dl, device)
            writer.add_scalar(f'AE/test/{test_loss.name}', test_loss.get(), epoch)
            writer.add_image('AE/test/image', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()

            # save the best autoencoder
            current_test_loss = test_loss.get()
            if current_test_loss < best_loss:
                torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(autoencoder)

    return best_model


def clustering_dl_latent(train_ds: Dataset, test_ds: Dataset, log_dir: str, args: Namespace):
    torch.manual_seed(RANDOM_SEED)

    # make dataset NxCxL
    shape = train_ds.signals.shape
    if len(shape) == 2:
        train_ds.signals = np.expand_dims(train_ds.signals, 1)
        test_ds.signals = np.expand_dims(test_ds.signals, 1)

    # train the autoencoder
    if args.load_path == "":
        autoencoder = train_autoencoder(train_ds, test_ds, log_dir, args)
    else:
        autoencoder = torch.load(args.load_path)

    # prepare a new dataset with latent representations
    with torch.no_grad():
        autoencoder = autoencoder.cpu()
        x_train = torch.Tensor(np.transpose(train_ds.signals, [0, 2, 1])).cpu()
        x_test = torch.Tensor(np.transpose(test_ds.signals, [0, 2, 1])).cpu()
        train_ds.signals = autoencoder.encoder(x_train)
        test_ds.signals = autoencoder.encoder(x_test)

    clustering_dl_raw(train_ds, test_ds, log_dir, args)
