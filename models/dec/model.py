import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from models import autoencoders


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
        dist = torch.linalg.norm(x.unsqueeze(1) - self.centroids, dim=-1, ord=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        return q / torch.sum(q, -1, keepdim=True)

    @staticmethod
    def set_random_centroids(num_centroids, size_centroids):
        centroids = nn.Parameter(torch.zeros([num_centroids, size_centroids]))
        nn.init.xavier_uniform_(centroids)
        return centroids

    @staticmethod
    def set_kmeans_centroids(embeddings, num_centroids):
        method = KMeans(n_clusters=num_centroids, n_init=1)
        method.fit_predict(embeddings)
        initial_centroids = method.cluster_centers_
        return nn.Parameter(torch.Tensor(initial_centroids))

    def setup(self, args):
        backprop_config = autoencoders.ops.BackpropConfig()
        backprop_config.optimizer = torch.optim.AdamW
        backprop_config.model = self
        backprop_config.lr = args.lr
        backprop_config.eta_min = args.eta_min
        backprop_config.epochs = args.epochs_dec
        backprop_config.weight_decay = args.weight_decay
        return autoencoders.ops.backprop_init(backprop_config)
