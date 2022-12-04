import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import seaborn as sns

from data import helpers

sns.set()

config_file = os.path.join(os.getcwd(), 'config', "put.yaml")
model_file = os.path.join(os.getcwd(), 'results', "put", "dl_raw", "Nov09_17-35-15_mbed", "best_clustering_model.pt")

with open(config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

total_dataset = helpers.load_dataset(config)
model = torch.load(model_file).cpu()

num_samples = total_dataset.signals.shape[-1]
num_modalities = total_dataset.signals.shape[-2]
num_centroids = total_dataset.num_classes

dataloader = DataLoader(total_dataset)

for CLASS in range(0, total_dataset.num_classes):

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))

    centroids = model.centroids
    # centroids = torch.reshape(model.centroids, (num_centroids, num_samples, num_modalities))
    centroids = centroids.detach().cpu().numpy()

    chosen_centroid = centroids[CLASS]
    t = np.linspace(0, 1, chosen_centroid.shape[0])

    for i, batch in enumerate(dataloader):
        x, y = batch

        pred = model(x.reshape(1, -1))
        y_hat = torch.argmax(pred, -1)

        if y_hat == CLASS:
            xx = x[0].numpy().transpose()
            xx = xx.reshape(-1)
            sns.lineplot(t, xx, ax=ax, color='blue', linewidth=0.5, alpha=0.2)

    sns.lineplot(t, chosen_centroid, ax=ax, color='red', linewidth=1.0)
    plt.show()
