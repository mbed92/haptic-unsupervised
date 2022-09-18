import torch
import torch.nn as nn

import utils.sklearn_benchmark
from utils.clustering import distribution_hardening
from utils.metrics import Mean


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

        # gather metrics
        x, y = outputs.detach().cpu().numpy(), batch_labels.detach().cpu().numpy()
        y_hat = torch.argmax(outputs, -1)

        # break when the training collapsed and all samples are in one cluster
        num_of_pred_clusters = torch.unique(y_hat).shape[0]
        if num_of_pred_clusters > 1:
            y_hat = y_hat.detach().cpu().numpy()
            for i in range(len(clustering_metrics_x_labels)):
                m = clustering_metrics_x_labels[i][1]

                # some metrics require 2 < num_of_pred_clusters < n_samples
                if 2 < num_of_pred_clusters < x.shape[0]:
                    metrics[i].add(m(x, y_hat))

            for i in range(len(clustering_metrics_true_pred)):
                m = clustering_metrics_true_pred[i][1]
                ii = len(clustering_metrics_x_labels) + i
                metrics[ii].add(m(y, y_hat))
        else:
            return None, None

    return clustering_loss, metrics
