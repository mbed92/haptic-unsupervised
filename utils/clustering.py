import io
import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from utils.metrics import clustering_accuracy


def save_embeddings(log_dir, embeddings: torch.Tensor, labels: torch.Tensor, writer: SummaryWriter,
                    global_step: int = 0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, f'metadata_{global_step}.tsv'), "w") as f:
        for label in labels:
            f.write(f'{label}\n')

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


def distribution_hardening(q):
    p = torch.div(q ** 2, torch.sum(q, 0, keepdim=True))
    return torch.div(p, torch.sum(p, 1, keepdim=True))


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
