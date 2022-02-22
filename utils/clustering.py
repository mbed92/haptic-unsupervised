import io
import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


def kmeans(x_train, x_test, expected_num_clusters):
    kmeans = KMeans(n_clusters=expected_num_clusters, n_init=20)
    return torch.Tensor(kmeans.fit_predict(x_train)), torch.Tensor(kmeans.predict(x_test))


def measure_clustering_accuracy(y_train, y_hat_train, y_test, y_hat_test):
    print('===================')
    print('| KMeans train accuracy:', clustering_accuracy(y_train, y_hat_train).numpy(),
          '| KMeans test accuracy:', clustering_accuracy(y_test, y_hat_test).numpy())
    print('===================')


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true, y_pred = y_true.int(), y_pred.int()

    # add 1 when classes are ordered from 0
    pred_idx_min, pred_idx_max = int(torch.min(y_pred).item()), int(torch.max(y_pred).item())
    true_idx_min, true_idx_max = int(torch.min(y_true).item()), int(torch.max(y_true).item())

    pred_idx = pred_idx_max + 1 if pred_idx_min == 0 else pred_idx_max
    true_idx = true_idx_max + 1 if true_idx_min == 0 else true_idx_max

    total = y_pred.shape[0]
    sample_idx = torch.arange(total)
    data = torch.zeros([total, pred_idx, true_idx], dtype=torch.float32)
    indices = torch.stack([sample_idx, y_pred, y_true], 1).type(torch.int64)
    indices_by_columns = [indices[:, i] for i in range(indices.shape[-1])]
    data[indices_by_columns] = 1.  # scatter_nd
    assignment_cost = torch.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind].float()  # gather_nd
    return torch.sum(gathered) / total


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


def create_img(arr1, arr2):
    plt.figure()

    series1 = [arr1[:, i] for i in range(arr1.shape[-1])]
    series2 = [arr2[:, i] for i in range(arr2.shape[-1])]
    t = np.arange(0, arr1.shape[0], 1)
    for s1, s2 in zip(series1, series2):
        plt.plot(t, s1, 'r')
        plt.plot(t, s2, 'g')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image = PIL.Image.open(buf)
    return ToTensor()(image)[:3]
