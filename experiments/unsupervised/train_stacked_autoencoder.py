import argparse
import os
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import submodules.haptic_transformer.utils as utils_haptr
import utils
from models.autoencoder import TimeSeriesAutoencoderConfig
from utils.dataset_loaders import EmbeddingDataset

torch.manual_seed(42)


def query(model, x):
    y_hat = model(x.permute(0, 2, 1)).permute(0, 2, 1)
    loss = nn.MSELoss()(y_hat, x)
    return y_hat, loss


def train_epoch(model, dataloader, optimizer, device):
    mean_loss = list()
    model.train(True)

    # loop over the epoch
    for data in dataloader:
        x, _ = data
        optimizer.zero_grad()
        y_hat, loss = query(model, x.to(device).float())
        loss.backward()
        optimizer.step()
        mean_loss.append(loss.item())

    return sum(mean_loss) / len(mean_loss)


def test_epoch(model, dataloader, device):
    mean_loss = list()
    exemplary_data = None
    model.train(False)

    # loop over the epoch
    with torch.no_grad():
        for data in dataloader:
            x, _ = data
            y_hat, loss = query(model, x.to(device).float())
            mean_loss.append(loss.item())

            # add the reconstruction result to the image
            if exemplary_data is None:
                exemplary_data = [y_hat[0].detach().cpu().numpy(), data[0][0].detach().cpu().numpy()]

    return sum(mean_loss) / len(mean_loss), exemplary_data


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # set up a model (find the best config)
    nn_params = TimeSeriesAutoencoderConfig()
    nn_params.data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    nn_params.stride = 2
    nn_params.kernel = 3
    nn_params.activation = nn.ReLU()
    nn_params.dropout = 0.2
    nn_params.num_heads = 1
    nn_params.use_attention = True
    autoencoder = models.TimeSeriesAutoencoder(nn_params)
    device = utils.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # start pretraining SAE auto encoders
    if args.pretrain_sae:
        train_dataloader = deepcopy(main_train_dataloader)
        test_dataloader = deepcopy(main_test_dataloader)

        for i, sae in enumerate(autoencoder.sae_modules):

            # setup backprop config for the full AE
            backprop_config_sae = utils.ops.BackpropConfig()
            backprop_config_sae.model = sae
            backprop_config_sae.lr = args.lr_sae
            backprop_config_sae.eta_min = args.eta_min_sae
            backprop_config_sae.epochs = args.epochs_sae
            backprop_config_sae.weight_decay = args.weight_decay_sae

            with SummaryWriter(log_dir=os.path.join(log_dir, f'sae{i}')) as writer:
                optimizer, scheduler = utils.ops.backprop_init(backprop_config_sae)
                sae.set_dropout(args.dropout)

                # run train/test epoch
                for epoch in range(args.epochs_sae):
                    train_epoch_loss = train_epoch(sae, train_dataloader, optimizer, device)
                    writer.add_scalar('loss/train/SAE', train_epoch_loss, epoch)
                    writer.add_scalar('lr/train/SAE', optimizer.param_groups[0]['lr'], epoch)
                    writer.flush()
                    scheduler.step()

                    # test epoch
                    test_epoch_loss, _ = test_epoch(sae, test_dataloader, device)
                    writer.add_scalar('loss/test/SAE', test_epoch_loss, epoch)
                    writer.flush()

                # prepare data for the previously trained SAE for the next SAE
                train_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, train_dataloader, device)
                test_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, test_dataloader, device)

    # train the main autoencoder
    backprop_config_ae = utils.ops.BackpropConfig()
    backprop_config_ae.model = autoencoder
    backprop_config_ae.lr = args.lr_ae
    backprop_config_ae.eta_min = args.eta_min_ae
    backprop_config_ae.epochs = args.epochs_ae
    backprop_config_ae.weight_decay = args.weight_decay_ae

    # train
    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = utils.ops.backprop_init(backprop_config_ae)

        for epoch in range(args.epochs_ae):
            train_epoch_loss = train_epoch(autoencoder, main_train_dataloader, optimizer, device)
            writer.add_scalar('loss/train/AE', train_epoch_loss, epoch)
            writer.add_scalar('lr/train/AE', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            scheduler.step()

            # test epoch
            test_epoch_loss, exemplary_sample = test_epoch(autoencoder, main_test_dataloader, device)
            writer.add_scalar('loss/test/AE', test_epoch_loss, epoch)
            writer.add_image('image/test/AE', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()

    # save the trained autoencoder
    torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))

    # verify the unsupervised classification accuracy
    with torch.no_grad():
        autoencoder.cpu()
        x_train, y_train = utils.dataset.get_total_data_from_dataloader(main_train_dataloader)
        x_test, y_test = utils.dataset.get_total_data_from_dataloader(main_test_dataloader)
        emb_train = autoencoder.encoder(x_train.permute(0, 2, 1)).numpy()
        emb_test = autoencoder.encoder(x_test.permute(0, 2, 1)).numpy()
        pred_train, pred_test = utils.clustering.kmeans(emb_train, emb_test, train_ds.num_classes)
        utils.clustering.measure_clustering_accuracy(y_train, pred_train, y_test, pred_test)

    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_train'), emb_train, y_train, writer)
    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/unsupervised/put.yaml")
    parser.add_argument('--epochs-sae', type=int, default=1)
    parser.add_argument('--epochs-ae', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--lr-sae', type=float, default=1e-3)
    parser.add_argument('--lr-ae', type=float, default=5e-4)
    parser.add_argument('--weight-decay-sae', type=float, default=1e-3)
    parser.add_argument('--weight-decay-ae', type=float, default=1e-3)
    parser.add_argument('--eta-min-sae', type=float, default=1e-4)
    parser.add_argument('--eta-min-ae', type=float, default=1e-4)
    parser.add_argument('--pretrain-sae', dest='pretrain_sae', action='store_true')
    parser.add_argument('--dont-pretrain-sae', dest='pretrain_sae', action='store_false')
    parser.set_defaults(pretrain_sae=True)

    args, _ = parser.parse_known_args()
    main(args)