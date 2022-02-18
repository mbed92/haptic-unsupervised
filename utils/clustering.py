import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    if type(y_true) is np.ndarray:
        y_true = torch.from_numpy(y_true)
    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred)

    y_true, y_pred = y_true.int(), y_pred.int()
    total = y_pred.shape[0]
    m = max(int(torch.max(y_pred).item()) + 1, int(torch.max(y_true).item()))
    indices = torch.stack([torch.arange(total), y_pred, y_true], 1).type(torch.int64)
    data = torch.zeros([total, m, m], dtype=torch.float32)
    indices_by_columns = [indices[:, i] for i in range(indices.shape[-1])]
    data[indices_by_columns] = 1.
    data = data.sum(0)
    data = torch.max(data) - data
    indices = linear_sum_assignment(data)
    indices = torch.stack([torch.Tensor(t) for t in indices], 1).type(torch.int64)
    return torch.sum(data[indices]) / total


def save_embeddings(log_dir, embeddings: torch.Tensor, labels: torch.Tensor, writer: SummaryWriter,
                    global_step: int = 0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, f'metadata_{global_step}.tsv'), "w") as f:
        for l in labels:
            f.write(f'{l}\n')

    torch.save({"embedding": embeddings}, os.path.join(log_dir, f"embeddings_{global_step}"))
    writer.add_embedding(embeddings, labels, global_step=global_step)
