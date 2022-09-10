import time
import warnings
from itertools import islice, cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import utils.sklearn_benchmark

sns.set()

RANDOM_SEED = 0
DEFAULT_PARAMS = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 2,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}
COLOR_BASE = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def setup_params(x):
    params = DEFAULT_PARAMS.copy()

    # normalize dataset for easier parameter selection
    x = StandardScaler().fit_transform(x)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(x, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        x, n_neighbors=params["n_neighbors"], include_self=False
    )

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    return params, bandwidth, connectivity


def cluster_raw_signals_ml(train_ds: Dataset, test_ds: Dataset):
    np.random.seed(RANDOM_SEED)
    total_dataset = train_ds + test_ds
    x, y = total_dataset.signals, total_dataset.labels

    # setup clustering algorithms
    params, bandwidth, connectivity = setup_params(x)
    clustering_algorithms = utils.sklearn_benchmark.get_sklearn_clustering_algorithms(params, bandwidth, connectivity)
    clustering_metrics = utils.sklearn_benchmark.get_sklearn_clustering_metrics()

    # setup matplotlib
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )
    rows = 4
    cols = int(np.ceil(len(clustering_algorithms) / rows))

    # start
    for plot_num, (algorithm_name, algorithm) in enumerate(clustering_algorithms):
        print(f"{algorithm_name} started...")

        # catch warnings related to kneighbors_graph
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                        + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(x)
        t1 = time.time()

        # get predictions
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(x)

        # append title
        plt.subplot(rows, cols, plot_num + 1)
        plt.title(algorithm_name, size=18)

        # show PCA
        pca = PCA(n_components=2)
        pca.fit(x)
        x_pca = pca.transform(x)

        # setup colors
        colors = np.array(list(islice(cycle(COLOR_BASE), int(max(y_pred) + 1), )))
        colors = np.append(colors, ["#000000"])

        # plot data
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.colorbar()

        print(f"{algorithm_name} finished in {t1 - t0}.")

        for sklearn_metric_name, sklearn_metric in clustering_metrics:
            print(f"{sklearn_metric_name} achieved {sklearn_metric(x, y_pred)}.")

    plt.show()
