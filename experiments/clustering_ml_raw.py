import os.path
import pickle
import time
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset

from utils.sklearn_benchmark import RANDOM_SEED, SCIKIT_LEARN_PARAMS, get_sklearn_clustering_algorithms, \
    get_sklearn_clustering_metrics_x_labels, get_sklearn_clustering_metrics_true_pred

sns.set()


# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def calculate_connectivity(x, params):
    # connectivity matrix for structured Ward and Agglomerative clustering
    connectivity = kneighbors_graph(x, n_neighbors=params["n_neighbors"])
    connectivity = 0.5 * (connectivity + connectivity.T)
    return connectivity


def clustering_ml_raw(total_dataset: Dataset, log_dir: str, expected_num_clusters: int):
    np.random.seed(RANDOM_SEED)
    x, y = total_dataset.signals, total_dataset.labels

    # flatten if needed
    preprocessing = False
    if len(x.shape) > 2:
        x = np.reshape(x, newshape=(x.shape[0], -1))
        preprocessing = True

    # setup clustering algorithms
    SCIKIT_LEARN_PARAMS["n_clusters"] = expected_num_clusters
    connectivity = calculate_connectivity(x, SCIKIT_LEARN_PARAMS)
    clustering_algorithms = get_sklearn_clustering_algorithms(SCIKIT_LEARN_PARAMS, connectivity)
    clustering_metrics_x_labels = get_sklearn_clustering_metrics_x_labels()
    clustering_metrics_true_pred = get_sklearn_clustering_metrics_true_pred()

    # setup matplotlib
    n_cols = np.ceil(len(clustering_algorithms) / SCIKIT_LEARN_PARAMS["n_rows"]).astype(np.int)
    fig, axs = plt.subplots(SCIKIT_LEARN_PARAMS["n_rows"], n_cols,
                            constrained_layout=True,
                            figsize=SCIKIT_LEARN_PARAMS["figsize"])

    # setup logdir
    log_file = os.path.join(log_dir, "log.txt")
    log_picture = os.path.join(log_dir, "tsne.png")

    # find TSNE mapping (one for all algorithms)
    tsne = TSNE(n_components=SCIKIT_LEARN_PARAMS["tsne_n_components"])
    x_tsne = tsne.fit_transform(x)

    # start benchmarking
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            for plot_num, (algorithm_name, algorithm) in enumerate(clustering_algorithms):
                if preprocessing:
                    steps = [('preprocessing', TruncatedSVD(SCIKIT_LEARN_PARAMS['svd_components'])),
                             ("inference", algorithm)]
                else:
                    steps = [("inference", algorithm)]

                pipeline = Pipeline(steps=steps)
                print(f"{algorithm_name} started... Number of steps: {len(steps)}")

                # inference the algorithm
                t0 = time.time()
                y_pred = pipeline.fit_predict(x)
                t1 = time.time()

                # print metrics
                print(f"{algorithm_name} finished in {t1 - t0}.")
                metrics = list()
                for sklearn_metric_name, sklearn_metric in clustering_metrics_x_labels:
                    value = sklearn_metric(x, y_pred)
                    metrics.append((sklearn_metric_name, value))
                    print(f"{sklearn_metric_name} achieved {value}.")
                for sklearn_metric_name, sklearn_metric in clustering_metrics_true_pred:
                    value = sklearn_metric(y, y_pred)
                    metrics.append((sklearn_metric_name, value))
                    print(f"{sklearn_metric_name} achieved {value}.")
                print("===========================\n\n")

                # TSNE
                colors = plt.cm.rainbow(np.linspace(0, 1, expected_num_clusters))
                ax = axs.reshape(-1)[plot_num]
                ax.set_title(algorithm_name, size=SCIKIT_LEARN_PARAMS["title_size"])
                ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)

                # save embeddings
                file_handler = open(os.path.join(log_dir, "".join((algorithm_name, ".pickle"))), "wb")
                pickle.dump({
                    "x_tsne": x_tsne,
                    "y_supervised": y,
                    "y_unsupervised": y_pred,
                    "metrics": metrics
                }, file_handler)

            # save tsne
            plt.savefig(log_picture, dpi=fig.dpi)
            plt.close(fig)
