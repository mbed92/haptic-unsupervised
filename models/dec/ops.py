import torch
import torch.nn as nn

import utils.sklearn_benchmark
from utils.clustering import distribution_hardening
from utils.metrics import Mean


def query(model, inputs, p_target):
    outputs = model(inputs)
    p_target = p_target.to(outputs.device).type(outputs.dtype)
    clustering_loss = nn.KLDivLoss(reduction="batchmean")(outputs.log(), p_target)
    return outputs, clustering_loss


def train(model, inputs, optimizer, p_target):
    optimizer.zero_grad()
    outputs, loss = query(model, inputs, p_target)
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
        batch_data, batch_labels, p_target = data
        outputs, loss = train(model, batch_data.to(device), optimizer, p_target)
        clustering_loss.add(loss.item())

        # gather metrics
        y_prob, y_true = outputs.detach().cpu().numpy(), batch_labels.detach().cpu().numpy()
        y_hat = torch.argmax(outputs, -1)

        # break when the training collapsed and all samples are in one cluster
        num_of_pred_clusters = torch.unique(y_hat).shape[0]
        if num_of_pred_clusters > 1:
            y_hat = y_hat.detach().cpu().numpy()

            # metric(x, y_hat)
            for i in range(len(clustering_metrics_x_labels)):
                m = clustering_metrics_x_labels[i][1]

                if 2 < num_of_pred_clusters < y_prob.shape[0]:
                    metrics[i].add(m(y_prob, y_hat))

            # metric(y, y_hat)
            for i in range(len(clustering_metrics_true_pred)):
                m = clustering_metrics_true_pred[i][1]
                ii = len(clustering_metrics_x_labels) + i
                metrics[ii].add(m(y_true, y_hat))

    return clustering_loss, metrics
