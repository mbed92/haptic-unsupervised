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

    if not {"x_tsne", "y_supervised", "y_unsupervised", "metrics"}.issubset(set(results.keys())):
        print(f"Bad keys: {results.keys()}.")
        return None

    return results


def plot_supervised_classes_in_unsupervised_clusters(title: str, data: dict, ax: plt.Axes):
    # assign separate color to each supervised class
    n_supervised_classes = int(max(data["y_supervised"]) + 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_supervised_classes))

    # plot TSNE results with supervised color-labels
    ax.set_title(title, size=DEFAULT_PARAMS["title_size"])
    ax.scatter(data["x_tsne"][:, 0], data["x_tsne"][:, 1],
               c=colors[data["y_supervised"]],
               edgecolor='none',
               alpha=0.5)

    # append centroids if possible
    if "centroids_tsne" in data.keys():
        ax.scatter(data["centroids_tsne"][:, 0], data["centroids_tsne"][:, 1],
                   c='black',
                   edgecolor='none',
                   alpha=0.5)


def print_supervised_classes_in_unsupervised_clusters(index_to_class: np.ndarray, results: dict):
    total_cluster_ids = np.unique(results["y_unsupervised"])
    _, cls_counts = np.unique(results["y_supervised"], return_counts=True)

    # iterate over existing clusters
    for cluster_id in total_cluster_ids:
        cluster_ids = np.argwhere(results["y_unsupervised"] == cluster_id)
        supervised_classes = results["y_supervised"][cluster_ids]
        supervised_classes, supervised_classes_counts = np.unique(supervised_classes, return_counts=True)
        supervised_classes_total_counts = cls_counts[supervised_classes]

        # turn indices into class names if possible
        if index_to_class is not None:
            supervised_classes = index_to_class[cluster_ids]
            supervised_classes = np.unique(supervised_classes)

        # print clustering info to the log file
        print(f"Cluster {cluster_id}:")
        for sc, scc, tot in zip(supervised_classes, supervised_classes_counts, supervised_classes_total_counts):
            print(f"{sc} - {scc} / {tot}")
        print("\n")


def analyze_clustering_results(dataset: Dataset, results_folder: str):
    dirs = glob.glob(os.path.join(results_folder, "*pickle"))

    # BAR plot
    bar_labels = list()
    bar_multi_heights, bar_multi_positions, bar_multi_widths, bar_multi_names = list(), list(), list(), list()

    # TSNE plot
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

        # 3. gather metrics for a bar plot (filter some of them)
        num_metrics = len(data["metrics"])
        bar_width = 1 / num_metrics - 0.05
        bars_width = num_metrics * bar_width
        bar_heights, bar_positions, bar_widths, bar_names = list(), list(), list(), list()

        for i, metric in enumerate(data["metrics"]):
            m_name, m_value = metric
            bar_names.append(m_name)
            bar_heights.append(m_value)
            bar_widths.append(bar_width)
            bar_positions.append(file_no - 0.5 * bars_width + i * bar_width)

        bar_labels.append(algorithm_name)
        bar_multi_names.append(bar_names)
        bar_multi_heights.append(bar_heights)
        bar_multi_widths.append(bar_widths)
        bar_multi_positions.append(bar_positions)

    # save the TSNE picture
    log_picture = os.path.join(results_folder, "summary_tsne.png")
    plt.savefig(log_picture, dpi=fig.dpi)
    plt.close(fig)

    # close previous figure and iterate again
    fig, ax = plt.subplots(constrained_layout=True, figsize=DEFAULT_PARAMS["figsize"])

    labels = bar_multi_names[0]
    for i, name in enumerate(labels):
        series_heights = [bmh[i] for bmh in bar_multi_heights]
        series_widths = [bmw[i] for bmw in bar_multi_widths]
        series_positions = [bmp[i] for bmp in bar_multi_positions]
        ax.bar(series_positions, series_heights, series_widths, label=name)

    ax.set_ylabel('Score', fontsize=DEFAULT_PARAMS["title_size"])
    ax.set_title('Metrics achieved by clustering methods', size=DEFAULT_PARAMS["title_size"])
    ax.set_xticks([sum(bp) / len(bp) for bp in bar_multi_positions], bar_labels)
    ax.legend(fontsize=DEFAULT_PARAMS["title_size"])

    log_picture = os.path.join(results_folder, "summary_bar_plot.png")
    plt.savefig(log_picture, dpi=fig.dpi)
    plt.show()
    plt.close(fig)
