import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import ticker
from torch.utils.data import DataLoader
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from data import helpers
from experiments.analyze_clustering_results import PUT_CLASSES
from utils.clustering import distribution_hardening
from utils.metrics import clustering_accuracy

sns.set()

config_file = os.path.join(os.getcwd(), 'config', "put.yaml")
model_file = os.path.join(os.getcwd(), 'results', "put", "dl_raw", "Nov09_17-35-15_mbed", "best_clustering_model.pt")

with open(config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

total_dataset = helpers.load_dataset(config)
model = torch.load(model_file)
model.eval()
model.cpu()

num_samples = total_dataset.signals.shape[-1]
num_modalities = total_dataset.signals.shape[-2]
num_centroids = total_dataset.num_classes

dataloader = DataLoader(total_dataset)

centroids = model.centroids.detach().cpu().numpy()
centroids = centroids.reshape([-1, num_modalities, num_samples])
centroids = (centroids * total_dataset.std) + total_dataset.mean
centroids = centroids.reshape([num_centroids, -1])

n_cols = 2
n_rows = np.ceil(num_centroids / n_cols).astype(np.int)
fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=(n_cols * 10, n_rows * 10))
axes = axes.reshape(-1)

for CLASS in range(0, total_dataset.num_classes):

    ax = axes[CLASS]
    chosen_centroid = centroids[CLASS]
    t = np.linspace(0, 1, chosen_centroid.shape[0])

    y_true_total = list()
    y_hat_total = list()
    for i, batch in enumerate(dataloader):
        x, y = batch

        feed = x.reshape(1, -1)
        pred = model(feed)
        pred = distribution_hardening(pred)
        y_hat = torch.argmax(pred, -1)

        if y_hat == CLASS:
            xx = (x[0] * total_dataset.std) + total_dataset.mean
            xx = xx.numpy().transpose()
            xx = xx.reshape(-1)
            sns.lineplot(x=t, y=xx, ax=ax, color='blue', linewidth=0.5, alpha=0.2)
            y_true_total.append(y.numpy()[0])
            y_hat_total.append(y_hat.numpy()[0])

    class_names, class_count = np.unique(y_true_total, return_counts=True)
    [print(PUT_CLASSES[cn], cc) for cn, cc in zip(class_names, class_count)]
    print(clustering_accuracy(np.asarray(y_true_total), np.asarray(y_hat_total)))
    print("=========")

    chosen_centroid = gaussian_filter1d(chosen_centroid, sigma=2)
    sns.lineplot(x=t, y=chosen_centroid, ax=ax, color='red', linewidth=3.0)

    ax.yaxis.grid(False)  # Hide the horizontal gridlines
    ax.xaxis.grid(True)  # Show the vertical gridlines
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0 / num_modalities))
    ax.set_xlim([0, 1])
    ax.set_ylim([-1300, 1300])

plt.savefig("summary.png", dpi=fig.dpi)
