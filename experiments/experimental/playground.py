'''
Load the best performing clustering model and verify what inter-classes features are shared among clusters
'''

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

import submodules.haptic_transformer.utils as utils_haptr
import utils

# CLASSES = ["Bubble", "Cardboard1", "Cardboard2", "Gum", "Leather", "Natural bag", "Plastic", "Sheet", "Sponge",
#            "Styrofoam", "Synth. Bag"]

CLASSES = ["Cardboard1", "Cardboard2", "Gum", "Leather", "Natural bag", "Plastic", "Sheet", "Sponge", "Styrofoam"]


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'clust_model')

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, _, test_ds = data.dataset.load_dataset(config)

    # load a model
    with torch.no_grad():
        clust_model = torch.load(args.load_path).cpu()
        x_train, y_train = torch.Tensor(train_ds.signals).permute(0, 2, 1), torch.Tensor(train_ds.labels)
        x_test, y_test = torch.Tensor(test_ds.signals).permute(0, 2, 1), torch.Tensor(test_ds.labels)

        # emb_train = clust_model.autoencoder.encoder(x_train.permute(0, 2, 1))
        emb_test = clust_model.autoencoder.encoder(x_test)
        # pred_train, pred_test = kmeans(emb_train, emb_test, train_ds.num_classes)
        # pred_train = clust_model.predict_class(x_train.permute(0, 2, 1)).type(torch.float32)
        pred_test = clust_model.predict_class(x_test).type(torch.float32)
        acc = utils.clustering.clustering_accuracy(y_test, pred_test)
        print(acc)

        with SummaryWriter(log_dir=log_dir) as writer:
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test_pred'), emb_test, pred_test, writer)
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test_true'), emb_test, y_test, writer, 1)

    x_test = x_test.numpy()
    y_true = y_test.numpy()
    emb = emb_test.numpy()
    y_hat = pred_test.numpy()
    cluster_centers = clust_model.centroids.detach().numpy()[None, ...]

    # get all indices of clusters
    classes = np.array(CLASSES)
    clusters = np.unique(y_hat)
    distances = list()
    for i, predicted_cluster_idx in enumerate(clusters):
        cluster_idxs = np.argwhere(y_hat == predicted_cluster_idx)
        cluster_idxs_true = y_true[cluster_idxs]
        true_labels_in_cluster, true_labels_in_cluster_cnt = np.unique(cluster_idxs_true, return_counts=True)

        # # distances to other clusters
        # embeddings = emb[cluster_idxs]
        # dist = np.linalg.norm(cluster_centers - embeddings, axis=-1).mean(0)
        # d = np.round(dist, 2)

        embeddings = emb[cluster_idxs]
        distances_emb = list()
        for j, other_predicted_cluster_idx in enumerate(clusters):
            other_cluster_idxs = np.argwhere(y_hat == other_predicted_cluster_idx)
            embeddings_other = emb[other_cluster_idxs]
            dist = np.linalg.norm(embeddings.mean(0) - embeddings_other.mean(0), axis=-1).mean()
            distances_emb.append(dist)
        distances.append(distances_emb)

        print(f"Cluster no. {predicted_cluster_idx} ---> {classes[true_labels_in_cluster.astype(int)]} - {true_labels_in_cluster_cnt} - {distances_emb}")

    print(distances)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/unsupervised/touching.yaml")
    parser.add_argument('--load-path', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/experiments/unsupervised/clust_model/Mar04_14-20-47_mbed/clustering_test_model")

    args, _ = parser.parse_known_args()
    main(args)
