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

torch.manual_seed(42)


def query(model, inputs, target_distribution):
    outputs = model(inputs)
    x = inputs.view(inputs.size(0), -1)
    reconstruction_loss = torch.mean(nn.MSELoss()(x, outputs['reconstruction']))
    clustering_loss = torch.mean(torch.kl_div(outputs['assignments'], target_distribution))
    return outputs, reconstruction_loss, clustering_loss


def train(model, inputs, target_distribution, optimizer):
    optimizer.zero_grad()
    outputs, reconstruction_loss, clustering_loss = query(model, inputs, target_distribution)
    loss = reconstruction_loss + clustering_loss
    loss.backward()
    optimizer.step()
    return outputs, reconstruction_loss, clustering_loss


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'clust_model')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils_haptr.dataset.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # setup a model
    clust_model = models.ClusteringModel(data_shape, args.embed_size, train_ds.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clust_model.from_pretrained(args.load_path, train_dataloader, device)
    summary(clust_model, input_size=data_shape)

    optimizer = torch.optim.AdamW(clust_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    # train the clust_model model
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(args.epochs):

            clust_model.train(True)
            for step, data in enumerate(train_dataloader):
                batch_data, batch_labels = utils_haptr.dataset.load_samples_to_device(data, device)
                p_train = clust_model.predict_soft_assignments(batch_data)
                y_hat, recon_loss, cluster_loss = train(clust_model, batch_data, p_train, optimizer)
                assignments = torch.argmax(y_hat['assignments'], -1)
                writer.add_scalar('loss/train/recon_loss', recon_loss.item(), epoch)
                writer.add_scalar('loss/train/cluster_loss', cluster_loss.item(), epoch)
                writer.add_scalar('loss/train/acc', utils.clustering.clustering_accuracy(batch_labels, assignments),
                                  epoch)
                scheduler.step()

            mean_loss = list()
            clust_model.train(False)
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data, batch_labels = utils_haptr.dataset.load_samples_to_device(data, device)
                    p_test = clust_model.predict_soft_assignments(batch_data)
                    y_hat, recon_loss, cluster_loss = query(clust_model, batch_data, p_test)
                    assignments = torch.argmax(y_hat['assignments'], -1)
                    writer.add_scalar('loss/test/recon_loss', recon_loss.item(), epoch)
                    writer.add_scalar('loss/test/cluster_loss', cluster_loss.item(), epoch)
                    writer.add_scalar('loss/test/acc', utils.clustering.clustering_accuracy(batch_labels, assignments),
                                      epoch)
            writer.flush()

        # save trained clust_model model
        torch.save(clust_model, os.path.join(writer.log_dir, 'clustering_test_model'))

    # verify the unsupervised classification accuracy
    x_train = torch.cat([y[0] for y in train_dataloader], 0).to(device).type(torch.float32)
    y_train = torch.cat([y[1] for y in train_dataloader], 0).to(device).type(torch.float32)
    emb_train = clust_model.autoencoder.encoder(x_train).to(device).type(torch.float32)

    x_test = torch.cat([y[0] for y in test_dataloader], 0).to(device).type(torch.float32)
    y_test = torch.cat([y[1] for y in test_dataloader], 0).to(device).type(torch.float32)
    emb_test = clust_model.autoencoder.encoder(x_test).to(device).type(torch.float32)

    kmeans = KMeans(n_clusters=train_ds.num_classes, n_init=20)
    pred_train = kmeans.fit_predict(emb_train.cpu().detach().numpy())
    pred_train = torch.Tensor(pred_train).to(device)
    pred_test = kmeans.predict(emb_test.cpu().detach().numpy())
    pred_test = torch.Tensor(pred_test).to(device)

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
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/experiments/autoencoder/Feb18_20-46-39_mbed/full/test_model")

    args, _ = parser.parse_known_args()
    main(args)
