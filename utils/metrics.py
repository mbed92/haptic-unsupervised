import torch
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape  # it has to be rectangular to make sense
    y_true = y_true.cpu().int()
    y_pred = y_pred.cpu().int()

    num_samples = y_pred.shape[0]
    shape = (num_samples, int(torch.max(y_pred).item()) + 1, int(torch.max(y_true).item()) + 1)  # assume ordered from 0
    sample_idx = torch.arange(num_samples).to(y_pred.device)
    indices = torch.stack([sample_idx, y_pred, y_true], 1)
    data = torch.zeros(shape, dtype=torch.float32)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0  # scatter_nd
    data = data.sum(0)
    assignment_cost = torch.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind].float()  # gather_nd
    return torch.sum(gathered) / num_samples
