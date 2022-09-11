import copy
import os
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from data import helpers
from models.autoencoders.conv import TimeSeriesConvAutoencoderConfig, TimeSeriesConvAutoencoder
from utils.ops import train_epoch, test_epoch


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
    device = utils.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # train the main autoencoder
    backprop_config = utils.ops.BackpropConfig()
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
        optimizer, scheduler = utils.ops.backprop_init(backprop_config)

        for epoch in range(args.epochs):
            train_loss = train_epoch(autoencoder, train_dl, optimizer, device)
            writer.add_scalar(f'AE/train/{train_loss.name}', train_loss.get(), epoch)
            writer.add_scalar(f'AE/train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            scheduler.step()

            # test epoch
            test_loss, exemplary_sample = test_epoch(autoencoder, test_dl, device)
            writer.add_scalar(f'AE/test/{test_loss.name}', test_loss.get(), epoch)
            writer.add_image('AE/test/image', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()

            # save the best autoencoder
            current_test_loss = test_loss.get()
            if current_test_loss < best_loss:
                torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(autoencoder)

    # verify the unsupervised classification accuracy
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()
            x_train, y_train = helpers.get_total_data_from_dataloader(train_dl)
            x_test, y_test = helpers.get_total_data_from_dataloader(test_dl)
            emb_train = best_model.encoder(x_train.permute(0, 2, 1)).numpy()
            emb_test = best_model.encoder(x_test.permute(0, 2, 1)).numpy()
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_train'), emb_train, y_train, writer)
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test'), emb_test, y_test, writer, 1)

    return best_model
