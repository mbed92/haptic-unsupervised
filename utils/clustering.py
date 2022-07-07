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
    kmeans = KMeans(n_clusters=expected_num_clusters, n_init=100, max_iter=500)
    train_result = torch.Tensor(kmeans.fit_predict(x_train))
    test_result = torch.Tensor(kmeans.predict(x_test))
    return train_result, test_result


def measure_clustering_accuracy(y_train, y_hat_train, y_test, y_hat_test):
    print('===================')
    print('| KMeans train accuracy:', clustering_accuracy(y_train, y_hat_train).numpy(),
          '| KMeans test accuracy:', clustering_accuracy(y_test, y_hat_test).numpy())
    print('===================')


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true, y_pred = y_true.int(), y_pred.int()
    num_samples = y_pred.shape[0]
    shape = (
        num_samples, int(torch.max(y_pred).item()) + 1, int(torch.max(y_true).item()) + 1)  # classes ordered from 0
    sample_idx = torch.arange(num_samples).to(y_pred.device)
    indices = torch.stack([sample_idx, y_pred, y_true], 1).type(torch.int64)
    data = torch.zeros(shape, dtype=torch.float32)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0  # scatter_nd
    data = data.sum(0)
    assignment_cost = torch.max(data) - data
    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    gathered = data[row_ind, col_ind].float()  # gather_nd
    return torch.sum(gathered) / num_samples


def save_embeddings(log_dir, embeddings: torch.Tensor, labels: torch.Tensor, writer: SummaryWriter,
                    global_step: int = 0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, f'metadata_{global_step}.tsv'), "w") as f:
        for label in labels: f.write(f'{label}\n')

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
    plt.close()
    buf.seek(0)
    image = PIL.Image.open(buf)
    return ToTensor()(image)[:3]
