from argparse import Namespace

import numpy as np
import seaborn as sns
from torch.utils.data import Dataset

from .benchmark import RANDOM_SEED
from .train_autoencoder import train_autoencoder
import torch

sns.set()


def clustering_dl_raw(train_ds: Dataset, test_ds: Dataset, log_dir: str, args: Namespace):
    torch.manual_seed(RANDOM_SEED)

    # train the autoencoder
    if args.load_path == "":
        autoencoder = train_autoencoder(train_ds, test_ds, log_dir, args)
    else:
        autoencoder = torch.load(args.load_path)

    # set up the model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # clust_model = ClusteringModel(DEFAULT_PARAMS["n_clusters"])
    # clust_model.from_pretrained(args.load_path, x_train, y_train, device)
    # summary(clust_model, input_size=data_shape)

    # # set up the optimization
    # train_dataloader = DataLoader(train_ds, batch_size=backprop_config.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_ds, batch_size=backprop_config.batch_size, shuffle=True)
    # optimizer, scheduler = utils.ops.backprop_init(backprop_config)

    # # setup matplotlib
    # n_rows = DEFAULT_PARAMS["n_rows"]
    # n_cols = np.ceil(len(clustering_algorithms) / n_rows).astype(np.int)
    # fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=(15, 15))
    #
    # # setup logdir
    # log_file = os.path.join(log_dir, "log.txt")
    # log_picture = os.path.join(log_dir, "tsne.png")
    #
    # # start benchmarking
    # with open(log_file, 'w') as f:
    #     with redirect_stdout(f):
    #         for plot_num, (algorithm_name, algorithm) in enumerate(clustering_algorithms):
    #             print(f"{algorithm_name} started...\n")
    #
    #             # inference the algorithm
    #             t0 = time.time()
    #             algorithm.fit(x)
    #             t1 = time.time()
    #
    #             # get predictions
    #             if hasattr(algorithm, "labels_"):
    #                 y_pred = algorithm.labels_.astype(int)
    #             else:
    #                 y_pred = algorithm.predict(x)
    #
    #             # setup colors
    #             colors = np.array(list(islice(cycle(COLOR_BASE), int(max(y_pred) + 1), )))
    #             colors = np.append(colors, ["#000000"])
    #
    #             # plot TSNE
    #             tsne = TSNE(n_components=2)
    #             x_tsne = tsne.fit_transform(x)
    #             ax = axs.reshape(-1)[plot_num]
    #             ax.set_title(algorithm_name, size=18)
    #             ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors[y_pred], edgecolor='none', alpha=0.5)
    #
    #             # save embeddings
    #             file_handler = open(os.path.join(log_dir, "".join((algorithm_name, ".pickle"))), "wb")
    #             pickle.dump({
    #                 "tsne": x_tsne,
    #                 "subervised_labels": y_pred
    #             }, file_handler)
    #
    #             # print metrics
    #             print(f"{algorithm_name} finished in {t1 - t0}.")
    #             for sklearn_metric_name, sklearn_metric in clustering_metrics_x_labels:
    #                 print(f"{sklearn_metric_name} achieved {sklearn_metric(x, y_pred)}.")
    #             for sklearn_metric_name, sklearn_metric in clustering_metrics_true_pred:
    #                 print(f"{sklearn_metric_name} achieved {sklearn_metric(y, y_pred)}.")
    #             print("===========================\n\n")
    #
    #         # save tsne
    #         plt.savefig(log_picture, dpi=fig.dpi)
    #         plt.show()
    #         plt.close(fig)
