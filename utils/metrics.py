import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def kmeans(x_train, x_test, expected_num_clusters):
    kmeans = KMeans(n_clusters=expected_num_clusters, n_init=100, max_iter=500)
    train_result = torch.Tensor(kmeans.fit_predict(x_train))
    test_result = torch.Tensor(kmeans.predict(x_test))
    return train_result, test_result


def print_clustering_accuracy(y_train, y_hat_train, y_test, y_hat_test):
    print('===================')
    print('| KMeans train accuracy:', clustering_accuracy(y_train, y_hat_train).numpy(),
          '| KMeans test accuracy:', clustering_accuracy(y_test, y_hat_test).numpy())
    print('===================')


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    # TODO: make this matrix rectangular
    y_true, y_pred = y_true.int(), y_pred.int()
    num_samples = y_pred.shape[0]
    shape = (
        num_samples, int(torch.max(y_pred).item()) + 1, int(torch.max(y_true).item()) + 1)  # classes ordered from 0
    sample_idx = torch.arange(num_samples).to(y_pred.device)
    indices = torch.stack([sample_idx, y_pred, y_true], 1)
    data = torch.zeros(shape, dtype=torch.float32)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0  # scatter_nd
    data = data.sum(0)
    assignment_cost = torch.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind].float()  # gather_nd
    return torch.sum(gathered) / num_samples
