import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def purity_score(labels_true: np.ndarray, labels_pred: np.ndarray):
    """
    Reference: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    To compute purity, each cluster is assigned to the class which is most frequent in the cluster,
    and then the accuracy of this assignment is measured by counting the number of correctly assigned samples
    and dividing by the number of samples. Bad clustering have purity values close to 0, a perfect clustering has
    a purity of 1. High purity is easy to achieve when the number of clusters is large - in particular, purity is 1
    if each document gets its own cluster. Thus, we cannot use purity to trade off the quality of the clustering
    against the number of clusters.
    """
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray):
    num_samples = labels_pred.shape[0]
    sample_idx = np.arange(num_samples)
    indices = np.stack([sample_idx, labels_pred, labels_true], 1).astype(np.int)
    shape = (num_samples, int(np.max(labels_pred).item()) + 1, int(np.max(labels_true).item()) + 1)  # ordered from 0
    data = np.zeros(shape)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0  # scatter_nd
    data = data.sum(0)
    assignment_cost = np.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind]  # gather_nd
    return np.sum(gathered) / num_samples


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
