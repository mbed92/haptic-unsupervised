import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import models
import submodules.haptic_transformer.utils as utils_haptr
import utils
from experiments.unsupervised.train_ae import create_img
from utils.clustering import clustering_accuracy, infer_kmeans

torch.manual_seed(42)


def query_clust(model, inputs, target_distribution):
    outputs = model(inputs.permute(0, 2, 1))
    reconstruction_loss = nn.MSELoss()(outputs['reconstruction'].permute(0, 2, 1), inputs)
    clustering_loss = nn.KLDivLoss(reduction='batchmean')(outputs['assignments'].log(), target_distribution)
    return outputs, reconstruction_loss, clustering_loss


def train_clust(model, inputs, target_distribution, optimizer):
    optimizer.zero_grad()
    outputs, reconstruction_loss, clustering_loss = query_clust(model, inputs, target_distribution)
    loss = reconstruction_loss + clustering_loss
    loss.backward()
    optimizer.step()
    return outputs, reconstruction_loss, clustering_loss


def train_epoch(model, dataloader, optimizer, device):
    mean_recon_loss, mean_cluster_loss, accuracy = list(), list(), list()
    model.train(True)
    for step, data in enumerate(dataloader):
        batch_data, batch_labels = utils_haptr.dataset.load_samples_to_device(data, device)
        p_train = model.predict_soft_assignments(batch_data.permute(0, 2, 1))
        y_hat, recon_loss, cluster_loss = train_clust(model, batch_data, p_train, optimizer)
        predictions = torch.argmax(y_hat['assignments'], 1)
        acc = utils.clustering.clustering_accuracy(batch_labels, predictions)
        mean_recon_loss.append(recon_loss.item())
        mean_cluster_loss.append(cluster_loss.item())
        accuracy.append(acc.item())

    return sum(mean_recon_loss) / len(mean_recon_loss), \
           sum(mean_cluster_loss) / len(mean_cluster_loss), \
           sum(accuracy) / len(accuracy)


def test_epoch(model, dataloader, device):
    mean_recon_loss, mean_cluster_loss, accuracy = list(), list(), list()
    model.train(False)
    exemplary_sample = None
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_data, batch_labels = utils_haptr.dataset.load_samples_to_device(data, device)
            p_test = model.predict_soft_assignments(batch_data.permute(0, 2, 1))
            y_hat, recon_loss, cluster_loss = query_clust(model, batch_data, p_test)
            predictions = torch.argmax(y_hat['assignments'], 1)
            acc = utils.clustering.clustering_accuracy(batch_labels, predictions)
            mean_recon_loss.append(recon_loss.item())
            mean_cluster_loss.append(cluster_loss.item())
            accuracy.append(acc.item())

            if exemplary_sample is None:
                x1 = batch_data[0].detach().cpu().numpy()
                x2 = y_hat['reconstruction'][0].detach().permute(1, 0).cpu().numpy()
                exemplary_sample = [x1, x2]

    return sum(mean_recon_loss) / len(mean_recon_loss), \
           sum(mean_cluster_loss) / len(mean_cluster_loss), \
           sum(accuracy) / len(accuracy), \
           exemplary_sample


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'clust_model')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clust_model = models.ClusteringModel(data_shape, args.embed_size, train_ds.num_classes, device)
    clust_model.from_pretrained(args.load_path, train_dataloader, device)
    summary(clust_model, input_size=data_shape)

    optimizer = torch.optim.AdamW(clust_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    # train the clust_model model
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(args.epochs):
            recon_loss, clust_loss, accuracy = train_epoch(clust_model, train_dataloader, optimizer, device)
            writer.add_scalar('loss/train/recon_loss', recon_loss, epoch)
            writer.add_scalar('loss/train/cluster_loss', clust_loss, epoch)
            writer.add_scalar('loss/train/accuracy', accuracy, epoch)
            scheduler.step()

            recon_loss, clust_loss, accuracy, exemplary_sample = test_epoch(clust_model, train_dataloader, device)
            writer.add_scalar('loss/test/recon_loss', recon_loss, epoch)
            writer.add_scalar('loss/test/cluster_loss', clust_loss, epoch)
            writer.add_scalar('loss/test/accuracy', accuracy, epoch)
            writer.add_image('image/test/reconstruction', create_img(*exemplary_sample), epoch)

        # save trained clust_model model
        torch.save(clust_model, os.path.join(writer.log_dir, 'clustering_test_model'))

    # verify the unsupervised classification accuracy
    emb_train, emb_test, y_train, y_test, x_train, x_test = infer_kmeans(clust_model.autoencoder.encoder,
                                                                         train_dataloader, test_dataloader,
                                                                         train_ds.num_classes, device)

    # verivy the classification accuracy
    y_hat_train = clust_model.predict_class(x_train.to(device))
    y_hat_test = clust_model.predict_class(x_test.to(device))
    print('| KMeans train accuracy:', clustering_accuracy(y_train.to(device), y_hat_train).numpy(),
          '| KMeans test accuracy:', clustering_accuracy(y_test.to(device), y_hat_test).numpy())

    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'visualization_test'), emb_train, y_train, writer)
    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'visualization_train'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/submodules/haptic_transformer/experiments/config/put_haptr_12.yaml")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/experiments/autoencoder/Feb20_11-51-52_mbed/full/test_model")

    args, _ = parser.parse_known_args()
    main(args)
