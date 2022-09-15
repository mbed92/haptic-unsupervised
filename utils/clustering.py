import io
import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


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
    p = torch.nan_to_num(torch.div(q ** 2, torch.sum(q, 1, keepdim=True)))
    return torch.nan_to_num(torch.div(p, torch.sum(p, 1, keepdim=True)))
