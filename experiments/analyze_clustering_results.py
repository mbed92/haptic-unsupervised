import glob
import os
import pickle
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import Dataset

from experiments.benchmark import DEFAULT_PARAMS

sns.set()


def open_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"No such file: {path}.")
        return None

    if type(results) is not dict:
        print(f"Bad file format: {type(results)}.")
        return None

    if not {"x_tsne", "y_supervised", "y_unsupervised"}.issubset(set(results.keys())):
        print(f"Bad keys: {results.keys()}.")
        return None

    return results


def plot_supervised_classes_in_unsupervised_clusters(title: str, data: dict, ax: plt.Axes):
    n_supervised_classes = int(max(data["y_supervised"]) + 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_supervised_classes))

    ax.set_title(title, size=DEFAULT_PARAMS["title_size"])
    ax.scatter(data["x_tsne"][:, 0], data["x_tsne"][:, 1],
               c=colors[data["y_supervised"]],
               edgecolor='none',
               alpha=0.5)

    if "centroids_tsne" in data.keys():
        ax.scatter(data["centroids_tsne"][:, 0], data["centroids_tsne"][:, 1],
                   c='black',
                   edgecolor='none',
                   alpha=0.5)


def print_supervised_classes_in_unsupervised_clusters(index_to_class: np.ndarray, results: dict):
    cluster_ids = np.unique(results["y_unsupervised"])

    for cluster_id in cluster_ids:
        cluster_indices = np.argwhere(results["y_unsupervised"] == cluster_id)

        if index_to_class is None:
            supervised_classes_in_cluster = results["y_supervised"][cluster_indices]
        else:
            supervised_classes_in_cluster = index_to_class[cluster_indices]

        supervised_classes, supervised_classes_counts = np.unique(supervised_classes_in_cluster, return_counts=True)

        print(f"Cluster {cluster_id}:")
        for sc, scc in zip(supervised_classes, supervised_classes_counts):
            print(f"{sc} - {scc}")
        print("\n")


def analyze_clustering_results(dataset: Dataset, results_folder: str):
    dirs = glob.glob(os.path.join(results_folder, "*pickle"))
    n_rows = DEFAULT_PARAMS["n_rows"]
    n_cols = np.ceil(len(dirs) / n_rows).astype(np.int)
    fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=DEFAULT_PARAMS["figsize"])

    for file_no, results_file in enumerate(sorted(dirs)):
        algorithm_name = results_file.split("/")[-1]
        algorithm_name = algorithm_name.rsplit(".")[0]

        # open the file
        data = open_pickle(results_file)
        if data is None:
            continue

        # 1. create a scatter plot of supervised labels in unsupervised clusters
        plot_supervised_classes_in_unsupervised_clusters(algorithm_name, data, axs.reshape(-1)[file_no])

        # 2. print clustering info: cluster no | num supervised classes | list supervised classes
        log_summary = os.path.join(results_folder, f"{algorithm_name}_summary.txt")

        # prepare class-to-index mapping
        index_to_class = None
        if hasattr(dataset, "meta") and dataset.meta.shape[0] == data["y_supervised"].shape[0]:
            index_to_class = dataset.meta[:, 0]

        with open(log_summary, 'w') as f:
            with redirect_stdout(f):
                print_supervised_classes_in_unsupervised_clusters(index_to_class, data)

    # save the picture
    log_picture = os.path.join(results_folder, "summary.png")
    plt.savefig(log_picture, dpi=fig.dpi)
    plt.show()
    plt.close(fig)
