import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import submodules.haptic_transformer.utils as utils_haptr
import utils
from data import helpers
from data.clustering_dataset import ClusteringDataset
from models.dec import ClusteringModel
from utils.clustering import distribution_hardening, create_img
from utils.metrics import clustering_accuracy, print_clustering_accuracy

torch.manual_seed(42)


def query_clust(model, inputs, target_distribution):
    outputs = model(inputs.permute(0, 2, 1))
    reconstruction_loss = nn.MSELoss()(outputs['reconstruction'].permute(0, 2, 1), inputs)

    log_probs_input = outputs['assignments'].log()
    clustering_loss = nn.KLDivLoss(reduction="batchmean")(log_probs_input, target_distribution)
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
        batch_data, batch_labels, target_probs = data
        y_hat, recon_loss, cluster_loss = train_clust(model, batch_data.to(device), target_probs.to(device), optimizer)
        predictions = torch.argmax(y_hat['assignments'], 1)
        acc = clustering_accuracy(batch_labels.to(device), predictions)
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
            batch_data, batch_labels, target_probs = data
            y_hat, recon_loss, cluster_loss = query_clust(model, batch_data.to(device), target_probs.to(device))
            predictions = torch.argmax(y_hat['assignments'], 1)
            acc = clustering_accuracy(batch_labels, predictions.cpu())
            mean_recon_loss.append(recon_loss.item())
            mean_cluster_loss.append(cluster_loss.item())
            accuracy.append(acc.item())

            if exemplary_sample is None:
                exemplary_sample = [batch_data[0].detach().cpu().numpy(),
                                    y_hat['reconstruction'][0].permute(1, 0).detach().cpu().numpy()]

    return sum(mean_recon_loss) / len(mean_recon_loss), \
           sum(mean_cluster_loss) / len(mean_cluster_loss), \
           sum(accuracy) / len(accuracy), \
           exemplary_sample


def update_distribution(model: nn.Module, dataloader: DataLoader, device):
    with torch.no_grad():
        x, y = list(), list()
        probs = list()
        for data in dataloader:
            inputs = data[0].permute(0, 2, 1).to(device).float()
            p = distribution_hardening(model.predict_soft_assignments(inputs))

            probs.append(p)
            x.append(data[0])
            y.append(data[1])

        probs = torch.cat(probs, 0)
        x = torch.cat(x, 0).float()
        y = torch.cat(y, 0)
        return ClusteringDataset.with_target_probs(x, y, probs, batch_size=dataloader.batch_size)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'clust_model')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = helpers.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # check performance for a different number of clusters
    x_train, y_train = helpers.get_total_data_from_dataloader(train_dataloader)
    x_test, y_test = helpers.get_total_data_from_dataloader(test_dataloader)

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clust_model = ClusteringModel(train_ds.num_classes, device)
    clust_model.from_pretrained(args.load_path, train_dataloader, device)
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
    best_clustering_accuracy = 0.0
    log_dir_new = utils_haptr.log.logdir_name(log_dir, f'num_clust_{train_ds.num_classes}')
    with SummaryWriter(log_dir=log_dir_new) as writer:
        for run in range(args.runs):

            # create a ClusteringDataset that consists of x, y, and soft assignments
            train_dataloader = update_distribution(clust_model, train_dataloader, device)
            test_dataloader = update_distribution(clust_model, test_dataloader, device)

            for epoch in range(args.epochs_per_run):
                step = run * args.epochs_per_run + epoch

                # train epoch
                recon_loss, clust_loss, accuracy = train_epoch(clust_model, train_dataloader, optimizer, device)
                writer.add_scalar('loss/train/recon_loss', recon_loss, step)
                writer.add_scalar('loss/train/cluster_loss', clust_loss, step)
                writer.add_scalar('loss/train/accuracy', accuracy, step)
                writer.add_scalar('lr/train', optimizer.param_groups[0]['lr'], step)
                writer.flush()

                # test epoch
                with torch.no_grad():
                    recon_loss, clust_loss, accuracy, exemplary_sample = test_epoch(clust_model,
                                                                                    test_dataloader,
                                                                                    device)
                    writer.add_scalar('loss/test/recon_loss', recon_loss, step)
                    writer.add_scalar('loss/test/cluster_loss', clust_loss, step)
                    writer.add_scalar('loss/test/accuracy', accuracy, step)
                    writer.add_image('image/test/reconstruction', create_img(*exemplary_sample), step)
                    writer.flush()

                    # save the best trained clust_model
                    if accuracy > best_clustering_accuracy:
                        torch.save(clust_model, os.path.join(writer.log_dir, 'clustering_test_model'))
                        best_clustering_accuracy = accuracy

            scheduler.step()

    # verify the accuracy after training
    with torch.no_grad():
        clust_model.cpu()
        pred_train = clust_model.predict_class(x_train.permute(0, 2, 1)).type(torch.float32)
        pred_test = clust_model.predict_class(x_test.permute(0, 2, 1)).type(torch.float32)
        print_clustering_accuracy(y_train, pred_train, y_test, pred_test)

        # save embeddings
        emb_train = clust_model.autoencoder.encoder(x_train.permute(0, 2, 1))
        emb_test = clust_model.autoencoder.encoder(x_test.permute(0, 2, 1))
        utils.clustering.save_embeddings(os.path.join(writer.log_dir, f'vis_train'), emb_train, y_train, writer)
        utils.clustering.save_embeddings(os.path.join(writer.log_dir, f'vis_test'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/touching.yaml")
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--epochs-per-run', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/experiments/autoencoder/touching/full/test_model")

    args, _ = parser.parse_known_args()
    main(args)
