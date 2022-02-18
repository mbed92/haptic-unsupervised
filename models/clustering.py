import torch
import torch.nn as nn
from sklearn.cluster import KMeans

import utils
from models import TimeSeriesAutoencoder


class ClusteringModel(nn.Module):

    def __init__(self, data_shape: list, embedding_size, num_clusters, alpha=1.0):
        super().__init__()

        self.num_clusters = num_clusters
        self.alpha = alpha
        self.autoencoder = TimeSeriesAutoencoder(data_shape, embedding_size)
        self.centroids = torch.rand([num_clusters, embedding_size])

    def forward(self, inputs):
        x = self.autoencoder.encoder(inputs)
        y = self.autoencoder.decoder(x)
        q = self.t_student(x)

        return {
            'assignments': q,
            'reconstruction': y,
        }

    def t_student(self, x):
        dist = torch.norm(x[:, None] - self.centroids.to(x.device), dim=-1)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        return q / torch.sum(q, -1, keepdim=True)

    def predict_soft_assignments(self, data):
        x = self.autoencoder.encoder(data)
        soft_assignments = self.t_student(x)
        return distribution_hardening(soft_assignments)

    def predict_class(self, data):
        return torch.argmax(self.predict_soft_assignments(data), -1)

    def from_pretrained(self, model_path, dataloader, device):
        # centroids initialization
        x = torch.cat([y[0] for y in dataloader], 0).to(device)
        y = torch.cat([y[1] for y in dataloader], 0).to(device)
        self.autoencoder = torch.load(model_path)
        flatten_data_shape = 1.0
        for s in x[0].shape:
            flatten_data_shape *= s

        # find latent vectors
        x = x.view(-1, int(flatten_data_shape)).to(device).type(torch.float32)
        x_emb = self.autoencoder.encoder(x)

        # taking the best centroid according to the accuracy
        initial_centroids, best_accuracy = None, None
        for i in range(20):
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=1)
            predictions = kmeans.fit_predict(x_emb.cpu().detach().numpy())
            predictions = torch.Tensor(predictions).to(device)
            initial_accuracy = utils.clustering.clustering_accuracy(y, predictions).numpy()

            if best_accuracy is None or initial_accuracy > best_accuracy:
                best_accuracy = initial_accuracy
                initial_centroids = kmeans.cluster_centers_

        # save the best centroids for optimization
        self.centroids = torch.Tensor(initial_centroids).type(torch.float32)


def distribution_hardening(q):
    ## P function (hardening)
    p = torch.div(q ** 2, torch.sum(q, 0, keepdim=True))
    return torch.div(p, torch.sum(p, 1, keepdim=True))
