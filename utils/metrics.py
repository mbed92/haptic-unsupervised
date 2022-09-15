import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def untorch(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def purity_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    To compute purity, each cluster is assigned to the class which is most frequent in the cluster,
    and then the accuracy of this assignment is measured by counting the number of correctly assigned samples
    and dividing by the number of samples. Bad clustering have purity values close to 0, a perfect clustering has
    a purity of 1. High purity is easy to achieve when the number of clusters is large - in particular, purity is 1
    if each document gets its own cluster. Thus, we cannot use purity to trade off the quality of the clustering
    against the number of clusters.
    """
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    A measure that allows us to make this tradeoff is normalized mutual information or NMI.
    """
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def rand_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    The Rand index gives equal weight to false positives and false negatives. Separating similar documents
    is sometimes worse than putting pairs of dissimilar documents in the same cluster. We can use the F measure
    to penalize false negatives more strongly than false positives by selecting a value $\beta > 1$,
    thus giving more weight to recall.
    """
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    return metrics.rand_score(y_true, y_pred)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
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


def clustering_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape  # it has to be rectangular to make sense
    y_true = untorch(y_true)
    y_pred = untorch(y_pred)
    return clustering_accuracy(y_true, y_pred)


class Mean:
    def __init__(self, name):
        self._data = list()
        self.name = name

    @staticmethod
    def mean(m: list):
        if len(m) > 0:
            return sum(m) / len(m)
        return None

    def add(self, value: float):
        self._data.append(value)

    def get(self):
        if len(self._data) > 0:
            return self.mean(self._data)
        return np.inf
