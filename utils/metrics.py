import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def untorch(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def purity_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape  # it has to be rectangular to make sense
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)

    num_samples = y_pred.shape[0]
    sample_idx = np.arange(num_samples)
    indices = np.stack([sample_idx, y_pred, y_true], 1).astype(np.int)
    shape = (num_samples, int(np.max(y_pred).item()) + 1, int(np.max(y_true).item()) + 1)  # assume ordered from 0
    data = np.zeros(shape)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0  # scatter_nd
    data = data.sum(0)
    assignment_cost = np.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind]  # gather_nd
    return np.sum(gathered) / num_samples
