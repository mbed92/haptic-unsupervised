import torch
import torch.nn as nn
from sklearn.cluster import KMeans

N_INITIAL_TRIALS = 30


class ClusteringModel(nn.Module):

    def __init__(self, num_clusters: int, latent_size: int, alpha: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_size = latent_size
        self.alpha = alpha
        self.centroids = self.set_random_centroids(num_clusters, latent_size)

    def forward(self, embeddings):
        return self.t_student(embeddings)

    def t_student(self, x):
        dist = torch.sum((x.unsqueeze(1) - self.centroids) ** 2, 2)
        q = 1.0 / (1.0 + (dist / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        a = torch.sum(q, 1, keepdim=True)
        return q / a

    @staticmethod
    def set_random_centroids(num_centroids, size_centroids):
        centroids = nn.Parameter(torch.zeros([num_centroids, size_centroids]))
        nn.init.xavier_uniform_(centroids)
        return centroids

    @staticmethod
    def set_kmeans_centroids(embeddings, num_centroids, size_centroids):
        method = KMeans(n_clusters=num_centroids)
        method.fit_predict(embeddings)
        initial_centroids = method.cluster_centers_
        return nn.Parameter(torch.Tensor(initial_centroids))
