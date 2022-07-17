import argparse
import copy
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
from utils.clustering import distribution_hardening, print_clustering_accuracy
from utils.metrics import clustering_accuracy, nmi_score, rand_score, purity_score, Mean

torch.manual_seed(42)


def query(model, inputs, target_distribution):
    outputs = model(inputs.permute(0, 2, 1))
    reconstruction_loss = nn.MSELoss()(outputs['reconstruction'].permute(0, 2, 1), inputs)
    log_probs_input = outputs['assignments'].log()
    clustering_loss = nn.KLDivLoss(reduction="batchmean")(log_probs_input, target_distribution)
    return outputs, (reconstruction_loss, clustering_loss)


def train(model, inputs, target_distribution, optimizer):
    optimizer.zero_grad()
    outputs, losses = query(model, inputs, target_distribution)
    loss = torch.sum(torch.stack(losses))
    loss.backward()
    optimizer.step()
    return outputs, losses


def train_epoch(model, dataloader, optimizer, device):
    reconstruction_loss, clustering_loss = Mean("Reconstruction Loss"), Mean("Clustering Loss")
    accuracy, purity, nmi, rand_index = Mean("Accuracy"), Mean("Purity"), Mean("NMI"), Mean("Random Index")
    model.train(True)

    for data in dataloader:
        batch_data, batch_labels, target_probs = data
        outputs, losses = train(model, batch_data.to(device), target_probs.to(device), optimizer)

        # gather losses
        reconstruction_loss.add(losses[0])
        clustering_loss.add(losses[1])

        # gather metrics
        predictions = torch.argmax(outputs['assignments'], 1)
        accuracy.add(clustering_accuracy(batch_labels, predictions))
        purity.add(purity_score(batch_labels, predictions))
        nmi.add(nmi_score(batch_labels, predictions))
        rand_index.add(rand_score(batch_labels, predictions))

    return (reconstruction_loss, clustering_loss), (accuracy, purity, nmi, rand_index)


def test_epoch(model, dataloader, device, add_exemplary_sample=True):
    reconstruction_loss, clustering_loss = Mean("Reconstruction Loss"), Mean("Clustering Loss")
    accuracy, purity, nmi, rand_index = Mean("Accuracy"), Mean("Purity"), Mean("NMI"), Mean("Random Index")
    exemplary_sample = None

    model.train(False)
    with torch.no_grad():
        for data in dataloader:
            batch_data, batch_labels, target_probs = data
            outputs, losses = query(model, batch_data.to(device), target_probs.to(device))

            # gather losses
            reconstruction_loss.add(losses[0].item())
            clustering_loss.add(losses[1].item())

            # gather metrics
            predictions = torch.argmax(outputs['assignments'], 1)
            accuracy.add(clustering_accuracy(batch_labels, predictions))
            purity.add(purity_score(batch_labels, predictions))
            nmi.add(nmi_score(batch_labels, predictions))
            rand_index.add(rand_score(batch_labels, predictions))

            # add an exemplary sample
            if add_exemplary_sample and exemplary_sample is None:
                y_pred = outputs['reconstruction'][0].permute(1, 0).detach().cpu().numpy()
                y_true = batch_data[0].detach().cpu().numpy()
                exemplary_sample = [y_pred, y_true]

    return (reconstruction_loss, clustering_loss), (accuracy, purity, nmi, rand_index), exemplary_sample


def add_soft_predictions(model: ClusteringModel, x_data: torch.Tensor, y_data: torch.Tensor,
                         device: torch.device, batch_size: int):
    model.train(False)
    model.cpu()  # remember that loading a whole dataset might plug the whole (GPU) RAM
    with torch.no_grad():
        x = x_data.permute(0, 2, 1).float()
        p_data = distribution_hardening(model.predict_soft_assignments(x))
        p_data.requires_grad = False
        model.to(device)
        return ClusteringDataset.with_target_probs(x_data, y_data, p_data, batch_size=batch_size)


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

    # get all the data for further calculations (remember that you always need to iterate over some bigger datasets)
    x_train, y_train = helpers.get_total_data_from_dataloader(train_dataloader)
    x_test, y_test = helpers.get_total_data_from_dataloader(test_dataloader)

    # setup a model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clust_model = ClusteringModel(train_ds.num_classes, device)
    clust_model.from_pretrained(args.load_path, x_train, y_train, device)
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
    best_model = None
    best_clustering_metric = 0.0
    with SummaryWriter(log_dir=log_dir) as writer:
        for run in range(args.runs):

            # create a ClusteringDataset that consists of x, y, and soft assignments
            train_dataloader = add_soft_predictions(clust_model, x_train, y_train, device, args.batch_size)
            test_dataloader = add_soft_predictions(clust_model, x_test, y_test, device, args.batch_size)

            for epoch in range(args.epochs_per_run):
                step = run * args.epochs_per_run + epoch

                # train epoch
                losses, metrics = train_epoch(clust_model, train_dataloader, optimizer, device)
                for loss in losses:
                    writer.add_scalar(f"CLUSTERING/train/{loss.name}", loss.get(), step)
                for metric in metrics:
                    writer.add_scalar(f"CLUSTERING/train/{metric.name}", metric.get(), step)
                writer.flush()

                # test epoch
                losses, metrics, exemplary_sample = test_epoch(clust_model, test_dataloader, device)
                for loss in losses:
                    writer.add_scalar(f"CLUSTERING/test/{loss.name}", loss.get(), step)
                for metric in metrics:
                    writer.add_scalar(f"CLUSTERING/test/{metric.name}", metric.get(), step)
                writer.flush()

                # save the best trained clust_model
                idx = [i for i in range(len(metrics)) if args.best_model_criterion in metrics[i].name.lower()]
                clustering_metric = metrics[idx[0]].get() if len(idx) == 1 else None
                if clustering_metric is not None and clustering_metric > best_clustering_metric:
                    torch.save(clust_model, os.path.join(writer.log_dir, 'clustering_test_model'))
                    best_clustering_metric = clustering_metric
                    best_model = copy.deepcopy(clust_model)

            scheduler.step()

    # verify the accuracy of the best model
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()
            pred_train = best_model.predict_class(x_train.permute(0, 2, 1))
            pred_test = best_model.predict_class(x_test.permute(0, 2, 1))
            print_clustering_accuracy(y_train, pred_train, y_test, pred_test)
            emb_train = best_model.autoencoder.encoder(x_train.permute(0, 2, 1))
            emb_test = best_model.autoencoder.encoder(x_test.permute(0, 2, 1))
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, f'vis_train'), emb_train, y_train, writer)
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, f'vis_test'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/touching.yaml")
    parser.add_argument('--best-model-criterion', type=str, default="accuracy")
    parser.add_argument('--runs', type=int, default=50)  # play with runs and epochs_per_run
    parser.add_argument('--epochs-per-run', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--load-path', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/experiments/autoencoder/touching/full/test_model")

    args, _ = parser.parse_known_args()
    main(args)
