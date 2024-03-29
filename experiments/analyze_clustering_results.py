import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.metrics.cluster import contingency_matrix

from utils.sklearn_benchmark import SCIKIT_LEARN_PARAMS

sns.set_style(style="white")

PUT_CLASSES = ['Art. grass', 'Rubber', 'Carpet', 'PVC', 'Ceramic', 'Foam', 'Sand', 'Rocks']
TOUCHING_CLASSES = ['Soft Card.', 'Hard Card.', 'Rubber', 'Leather', 'Linen bag', 'Plastic', 'Steel', 'Sponge',
                    'Styro.', 'Plastic bag']
BIOTAC_CLASSES = ['Aluminum', 'Apple', 'Aquarius Can', 'Plastic Budha', 'Bunny Mascot', 'Pen Case', 'Coke Can',
                  'Toothpaste', 'Plastic Cone', 'Cookies Plastic', 'Plastic Creeper', 'Detergent', 'Egg Box',
                  'Plastic Elephant', 'Ball Mascot', 'Glass Bottle', 'Green Gel', 'Green Glass', 'Heat Sink',
                  'Juice Cartoon',
                  'Water Bottle', 'Lindor Box', 'Metal Box', 'Metal Pen Case', 'Milk Cartoon', 'Minion Mascot', 'Mug',
                  'Olaf Mascot', 'Pen Box', 'Plastic Ball', 'Pringles', 'Puleva Bottle', 'Red Can', 'Rollon',
                  'Rugby Ball', 'Salt Box', 'Food Can', 'Shampoo', 'Shoe', 'Shower Gel', 'Spray', 'Tape Measure',
                  'Taz Mascot', 'Tennis Ball', 'Toilet Paper', 'Toy Horn', 'Plastic Train', 'Twisted Plastic',
                  'Wooden Wardrobe', 'Wood Box', 'Yellow Sponge']

EXCLUDED_METRICS = [
    "AdjustedRand",
    "Silhouette"
]


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


def plot_tnse(title: str, x_tsne: np.ndarray, y_tsne: np.ndarray, predictions: np.ndarray, ax: plt.Axes):
    # assign separate color to each supervised class
    n_classes = int(max(predictions) + 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    # plot TSNE results with supervised color-labels
    ax.set_title(title, size=SCIKIT_LEARN_PARAMS["title_size"])
    ax.scatter(x_tsne, y_tsne, c=colors[predictions], edgecolor='none', alpha=0.5)


def log_info(index_to_class: np.ndarray, results: dict):
    total_cluster_ids = np.unique(results["y_unsupervised"])
    tot_cls_ids, tot_cls_idx, tot_cls_counts = np.unique(results["y_supervised"], return_counts=True, return_index=True)
    tot_cls_names = None
    if index_to_class is not None:
        tot_cls_names = index_to_class[tot_cls_idx]

    # iterate over existing clusters
    for cluster_id in total_cluster_ids:
        cluster_ids = np.argwhere(results["y_unsupervised"] == cluster_id)[:, 0]
        in_cls = results["y_supervised"][cluster_ids]
        in_cluster_cls = np.unique(in_cls)

        # fill with zeros for no classes inside
        classes_counts = list()
        i = 0
        for cls_id in tot_cls_ids:
            if cls_id in in_cluster_cls:
                num = len(np.where(in_cls == cls_id)[0])
                classes_counts.append(num)
            else:
                classes_counts.append(0)
            i += 1

        # print clustering info to the log file
        print(f"Cluster {cluster_id}")
        for sc, scc, tot in zip(tot_cls_names, classes_counts, tot_cls_counts):
            if scc > 0:
                print(f"{sc} - {((100 * scc) / tot):02f}%")
        print("\n")


def get_distance_mat(centroids: np.ndarray):
    return np.linalg.norm(centroids[np.newaxis, ...] - centroids[:, np.newaxis], axis=-1, keepdims=True)[..., 0]


def analyze_clustering_results(results_folder: str):
    f = os.path.join(results_folder, "**", "*.pickle")
    dirs = glob.glob(f, recursive=True)

    # TSNE plots
    n_rows = SCIKIT_LEARN_PARAMS["n_rows"]
    n_cols = np.ceil(len(dirs) / n_rows).astype(np.int)

    # create a space for the supervised classes TSNE
    if n_cols * n_rows == len(dirs):
        n_rows += 1

    # BAR plot
    bar_labels = list()
    bar_multi_heights, bar_multi_positions, bar_multi_widths, bar_multi_names = list(), list(), list(), list()

    # Contingency matrices
    cm_cond_list = list()
    algos_names = list()

    # generate data
    fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=SCIKIT_LEARN_PARAMS["figsize"])
    for file_no, results_file in enumerate(sorted(dirs)):
        algorithm_name = results_file.split("/")[-1].rsplit(".")[0]
        algos_names.append(algorithm_name)

        # open the file
        data = open_pickle(results_file)
        if data is None:
            continue

        # choose classes for the dataset
        if "put" in results_folder.lower():
            index_to_class = PUT_CLASSES
        elif "touching" in results_folder.lower():
            index_to_class = TOUCHING_CLASSES
        elif "biotac" in results_folder.lower():
            index_to_class = BIOTAC_CLASSES
        else:
            index_to_class = None

        # 1. create a scatter plot of supervised labels in unsupervised clusters
        # pick the same 2D TSNE for all plots (assume all results are about the same dataset)
        plot_tnse(algorithm_name, data["x_tsne"][:, 0], data["x_tsne"][:, 1],
                  data["y_unsupervised"], axs.reshape(-1)[file_no])

        # add reference TSNE with supervised classes, add it to the next Axis
        if file_no == len(dirs) - 1:
            plot_tnse("Supervised classes", data["x_tsne"][:, 0], data["x_tsne"][:, 1],
                      data["y_supervised"], axs.reshape(-1)[-1])

        # 2. gather metrics for a bar plot (filter some of them)
        metrics = list()
        for i in range(len(data["metrics"])):
            if data["metrics"][i][0] not in EXCLUDED_METRICS:
                metrics.append(data["metrics"][i])

        num_metrics = len(metrics)
        bar_width = 1 / num_metrics - 0.05
        bars_width = num_metrics * bar_width
        bar_heights, bar_positions, bar_widths, bar_names = list(), list(), list(), list()

        for i, metric in enumerate(metrics):
            m_name, m_value = metric
            bar_names.append(m_name)
            bar_heights.append(m_value)
            bar_widths.append(bar_width)
            bar_positions.append(file_no - 0.5 * bars_width + i * bar_width)

        if "Agglomerative" in algorithm_name:
            algorithm_name = "Agglomerative"
        if "Spectral" in algorithm_name:
            algorithm_name = "Spectral"

        bar_labels.append(algorithm_name)
        bar_multi_names.append(bar_names)
        bar_multi_heights.append(bar_heights)
        bar_multi_widths.append(bar_widths)
        bar_multi_positions.append(bar_positions)

        # 3. Contingency matrix
        cm = contingency_matrix(data["y_unsupervised"], data["y_supervised"])
        cm_cond = cm / cm.sum(axis=0)
        cm_cond_list.append(cm_cond)

        # save contingency matrix as csv using pandas
        if index_to_class is not None:
            df = pd.DataFrame(cm_cond, index=index_to_class)
            df.to_csv(os.path.join(results_folder, f"{algorithm_name}_contingency_matrix.csv"))

    # save the TSNE picture
    log_picture = os.path.join(results_folder, "summary_tsne.png")
    plt.savefig(log_picture, dpi=fig.dpi)
    plt.close(fig)

    # close previous figure and create a new with a bar plot
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
    labels = bar_multi_names[0]

    # save algos_names, labels, bar_multi_heights as csv using pandas
    df = pd.DataFrame(bar_multi_heights, index=algos_names, columns=labels)
    df.to_csv(os.path.join(results_folder, "summary_bar_plot.csv"))

    # plot the bar plot
    for i, name in enumerate(labels):
        series_heights = [bmh[i] for bmh in bar_multi_heights]
        series_widths = [bmw[i] for bmw in bar_multi_widths]
        series_positions = [bmp[i] for bmp in bar_multi_positions]
        ax.bar(series_positions, series_heights, series_widths, label=name)

    # set plot parameters
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    ax.set_ylabel('Score', fontsize=SCIKIT_LEARN_PARAMS["title_size"])
    ax.set_title('Metrics achieved by clustering methods', size=SCIKIT_LEARN_PARAMS["title_size"])
    ax.set_xticks([sum(bp) / len(bp) for bp in bar_multi_positions], bar_labels)
    ax.legend(fontsize=SCIKIT_LEARN_PARAMS["title_size"])
    log_picture = os.path.join(results_folder, "summary_bar_plot.png")
    plt.savefig(log_picture, dpi=fig.dpi)
    plt.close(fig)

    print("Done with bar plot.")
