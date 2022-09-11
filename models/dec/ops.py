import torch
import torch.nn as nn

import utils.sklearn_benchmark
from data.clustering_dataset import ClusteringDataset
from utils.clustering import distribution_hardening
from utils.metrics import Mean
from .model import ClusteringModel


def query(model, inputs):
    outputs = model(inputs)
    target = distribution_hardening(outputs).detach()
    clustering_loss = nn.KLDivLoss(reduction="batchmean")(outputs.log(), target)
    return outputs, clustering_loss


def train(model, inputs, optimizer):
    optimizer.zero_grad()
    outputs, loss = query(model, inputs)
    loss.backward()
    optimizer.step()
    return outputs, loss


def train_epoch(model, dataloader, optimizer, device):
    clustering_loss = Mean("Clustering Loss")
    clustering_metrics_x_labels = utils.sklearn_benchmark.get_sklearn_clustering_metrics_x_labels()
    clustering_metrics_true_pred = utils.sklearn_benchmark.get_sklearn_clustering_metrics_true_pred()
    metrics = [Mean(m[0]) for m in (clustering_metrics_x_labels + clustering_metrics_true_pred)]

    model.train(True)
    for data in dataloader:
        batch_data, batch_labels = data
        outputs, loss = train(model, batch_data.to(device), optimizer)
        clustering_loss.add(loss.item())

        x, y = outputs.detach().cpu().numpy(), batch_labels.detach().cpu().numpy()
        y_hat = torch.argmax(outputs, -1).detach().cpu().numpy()
        for i in range(len(clustering_metrics_x_labels)):
            m = clustering_metrics_x_labels[i][1]
            metrics[i].add(m(x, y))

        for i in range(len(clustering_metrics_true_pred)):
            m = clustering_metrics_true_pred[i][1]
            ii = len(clustering_metrics_x_labels) + i
            metrics[ii].add(m(y, y_hat))

    return clustering_loss, metrics


def add_soft_predictions(model: ClusteringModel,
                         x_data: torch.Tensor, y_data: torch.Tensor,
                         device: torch.device, batch_size: int):
    model.cpu()  # remember that loading a whole dataset might plug the whole (GPU) RAM
    model.train(False)
    with torch.no_grad():
        p_data = distribution_hardening(model(x_data))

    model.to(device)
    return ClusteringDataset.with_target_probs(x_data, y_data, p_data, batch_size=batch_size)

# def test_epoch(model, dataloader, device):
#     reconstruction_loss, clustering_loss = Mean("Reconstruction Loss"), Mean("Clustering Loss")
#     # accuracy, purity, nmi, rand_index = Mean("Accuracy"), Mean("Purity"), Mean("NMI"), Mean("Random Index")
#     exemplary_sample = None
#
#     model.train(False)
#     with torch.no_grad():
#         for data in dataloader:
#             batch_data, batch_labels, target_probs = data
#             outputs, loss = query(model, batch_data.to(device), target_probs.to(device))
#             clustering_loss.add(loss.item())
#
#     return clustering_loss, exemplary_sample
