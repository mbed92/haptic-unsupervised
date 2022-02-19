import argparse
import os

import torch
import torch.nn as nn
import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import models
import submodules.haptic_transformer.utils as utils_haptr
import utils
from utils.embedding_dataset import EmbeddingDataset

torch.manual_seed(42)


def query_ae(model, data, criterion):
    y_hat = model(data)
    loss = criterion(y_hat, data)
    return y_hat, loss


def train_ae(model, data, criterion, optimizer):
    optimizer.zero_grad()
    y_hat, loss = query_ae(model, data, criterion)
    loss.backward()
    optimizer.step()
    return y_hat, loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    mean_loss = list()
    model.train(True)
    for step, data in enumerate(dataloader):
        batch_data = data[0].view(data[0].size(0), -1).to(device).float()
        y_hat, loss = train_ae(model, batch_data, criterion, optimizer)
        mean_loss.append(loss.item())
    return sum(mean_loss) / len(mean_loss)


def test_epoch(model, dataloader, criterion, device):
    mean_loss = list()
    model.train(False)
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_data = data[0].view(data[0].size(0), -1).to(device).float()
            y_hat, loss = query_ae(model, batch_data, criterion)
            mean_loss.append(loss.item())
    return sum(mean_loss) / len(mean_loss)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils_haptr.dataset.load_dataset(config)
    train_ds += val_ds
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # setup a model
    autoencoder = models.TimeSeriesAutoencoder(data_shape, args.embed_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    summary(autoencoder, input_size=data_shape)

    # start pretraining SAE autoencoders
    if args.pretrain_sae:
        train_dataloader = main_train_dataloader
        test_dataloader = main_test_dataloader

        for i, sae in enumerate([autoencoder.sae1, autoencoder.sae2, autoencoder.sae3, autoencoder.sae4]):
            sae_log_dir = os.path.join(log_dir, f'sae{i}')
            sae.set_dropout(args.dropout)

            with SummaryWriter(log_dir=sae_log_dir) as writer:
                optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_sae,
                                                                       eta_min=args.eta_min)
                criterion = nn.MSELoss()

                # run train/test epoch
                for epoch in range(args.epochs_sae):
                    # train epoch
                    train_epoch_loss = train_epoch(sae, train_dataloader, optimizer, criterion, device)
                    writer.add_scalar(f'loss/train/SAE{i}', train_epoch_loss, epoch)
                    writer.add_scalar(f'lr/train/SAE{i}', optimizer.param_groups[0]['lr'], epoch)
                    scheduler.step()

                    # test epoch
                    test_epoch_loss = test_epoch(sae, test_dataloader, criterion, device)
                    writer.add_scalar(f'loss/test/SAE{i}', test_epoch_loss, epoch)
                    writer.flush()

                # prepare data for the previously trained SAE for the next SAE
                train_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, train_dataloader, device)
                test_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, test_dataloader, device)

    # train the main autoencoder
    main_log_dir = os.path.join(log_dir, f'full')
    with SummaryWriter(log_dir=main_log_dir) as writer:
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_autoencoder,
                                                               eta_min=args.eta_min)
        criterion = nn.MSELoss()

        for epoch in range(args.epochs_autoencoder):
            train_epoch_loss = train_epoch(autoencoder, main_train_dataloader, optimizer, criterion, device)
            writer.add_scalar('loss/train/AE', train_epoch_loss, epoch)
            scheduler.step()

            # test epoch
            test_epoch_loss = test_epoch(autoencoder, main_test_dataloader, criterion, device)
            writer.add_scalar('loss/test/AE', test_epoch_loss, epoch)
            writer.flush()

    # save the trained autoencoder
    torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))

    # verify the unsupervised classification accuracy
    x_train = torch.cat([y[0] for y in main_train_dataloader], 0).type(torch.float32)
    y_train = torch.cat([y[1] for y in main_train_dataloader], 0).type(torch.float32)
    emb_train = autoencoder.encoder(x_train.to(device)).detach().type(torch.float32)

    x_test = torch.cat([y[0] for y in main_test_dataloader], 0).type(torch.float32)
    y_test = torch.cat([y[1] for y in main_test_dataloader], 0).type(torch.float32)
    emb_test = autoencoder.encoder(x_test.to(device)).detach().type(torch.float32)

    kmeans = KMeans(n_clusters=train_ds.num_classes, n_init=20)
    pred_train = torch.Tensor(kmeans.fit_predict(emb_train.cpu().numpy()))
    pred_test = torch.Tensor(kmeans.predict(emb_test.cpu().numpy()))

    print('===================')
    print('| KMeans train accuracy:', utils.clustering.clustering_accuracy(y_train, pred_train).numpy(),
          '| KMeans test accuracy:', utils.clustering.clustering_accuracy(y_test, pred_test).numpy())
    print('===================')
    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'visualization_test'), emb_train, y_train, writer)
    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'visualization_train'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/submodules/haptic_transformer/experiments/config/put_haptr_12.yaml")
    parser.add_argument('--epochs-sae', type=int, default=400)
    parser.add_argument('--epochs-autoencoder', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--pretrain-sae', dest='pretrain_sae', action='store_true')
    parser.add_argument('--dont-pretrain-sae', dest='pretrain_sae', action='store_false')
    parser.set_defaults(pretrain_sae=True)

    args, _ = parser.parse_known_args()
    main(args)
