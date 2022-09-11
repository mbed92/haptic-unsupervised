import os.path
import pickle
import time
from contextlib import redirect_stdout
from itertools import islice, cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import utils.sklearn_benchmark
from .benchmark import RANDOM_SEED, DEFAULT_PARAMS, COLOR_BASE

sns.set()


def setup_params(x, params):
    # normalize dataset for easier parameter selection
    x = StandardScaler().fit_transform(x)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        x, n_neighbors=params["n_neighbors"], include_self=False
    )

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    return x, connectivity


def clustering_ml_raw(total_dataset: Dataset, log_dir: str):
    np.random.seed(RANDOM_SEED)
    x, y = total_dataset.signals, total_dataset.labels

    # flatten if needed
    if len(x.shape) > 2:
        x = np.reshape(x, newshape=(x.shape[0], -1))

    # setup clustering algorithms
    x, connectivity = setup_params(x, DEFAULT_PARAMS)
    clustering_algorithms = utils.sklearn_benchmark.get_sklearn_clustering_algorithms(DEFAULT_PARAMS, connectivity)
    clustering_metrics_x_labels = utils.sklearn_benchmark.get_sklearn_clustering_metrics_x_labels()
    clustering_metrics_true_pred = utils.sklearn_benchmark.get_sklearn_clustering_metrics_true_pred()

    # setup matplotlib
    n_rows = DEFAULT_PARAMS["n_rows"]
    n_cols = np.ceil(len(clustering_algorithms) / n_rows).astype(np.int)
    fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=(15, 15))

    # setup logdir
    log_file = os.path.join(log_dir, "log.txt")
    log_picture = os.path.join(log_dir, "tsne.png")

    # start benchmarking
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            for plot_num, (algorithm_name, algorithm) in enumerate(clustering_algorithms):
                print(f"{algorithm_name} started...\n")

                # inference the algorithm
                t0 = time.time()
                algorithm.fit(x)
                t1 = time.time()

                # get predictions
                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(x)

                # setup colors
                colors = np.array(list(islice(cycle(COLOR_BASE), int(max(y_pred) + 1), )))
                colors = np.append(colors, ["#000000"])

                # plot TSNE
                tsne = TSNE(n_components=2)
                x_tsne = tsne.fit_transform(x)
                ax = axs.reshape(-1)[plot_num]
                ax.set_title(algorithm_name, size=18)
                ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)

                # save embeddings
                file_handler = open(os.path.join(log_dir, "".join((algorithm_name, ".pickle"))), "wb")
                pickle.dump({
                    "tsne": x_tsne,
                    "subervised_labels": y_pred
                }, file_handler)

                # print metrics
                print(f"{algorithm_name} finished in {t1 - t0}.")
                for sklearn_metric_name, sklearn_metric in clustering_metrics_x_labels:
                    print(f"{sklearn_metric_name} achieved {sklearn_metric(x, y_pred)}.")
                for sklearn_metric_name, sklearn_metric in clustering_metrics_true_pred:
                    print(f"{sklearn_metric_name} achieved {sklearn_metric(y, y_pred)}.")
                print("===========================\n\n")

            # save tsne
            plt.savefig(log_picture, dpi=fig.dpi)
            plt.show()
            plt.close(fig)
