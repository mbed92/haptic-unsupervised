import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from utils.metrics import purity_score

N_INITIAL_TRIALS = 30


class ClusteringModel(nn.Module):

    def __init__(self, num_clusters: int, alpha=1.0):
        super().__init__()

        self.num_clusters = num_clusters
        self.alpha = alpha
        self.autoencoder = None
        self.centroids = self.set_random_centroids(self.num_clusters, 10)

    def forward(self, inputs):
        x = self.autoencoder.encoder(inputs)
        q = self.t_student(x)
        y = self.autoencoder.decoder(x)

        return {
            'assignments': q,
            'reconstruction': y,
        }

    def t_student(self, x):
        # dist = x[:, None] - self.centroids[None, :]
        # dist = torch.linalg.norm(dist, dim=-1)
        dist = torch.sum((x.unsqueeze(1) - self.centroids) ** 2, 2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        a = torch.sum(q, 1, keepdim=True)
        return q / a

    def predict_soft_assignments(self, data):
        return self.t_student(self.autoencoder.encoder(data))

    def predict_class(self, data):
        return torch.argmax(self.predict_soft_assignments(data), -1)

    def from_pretrained(self, model_path: str, input_samples: torch.Tensor, true_labels: torch.Tensor,
                        device: torch.device, best_centroids: bool = True):

        # centroids initialization
        self.autoencoder = torch.load(model_path).cpu()

        # fit the best centroids or set randomly
        inputs = input_samples.permute(0, 2, 1).float()
        if best_centroids:
            self.centroids = self.set_kmeans_centroids(self.autoencoder.encoder, inputs, true_labels, self.num_clusters)
        else:
            size_centroids = self.autoencoder.encoder(inputs).shape[-1]
            self.centroids = self.set_random_centroids(self.num_clusters, size_centroids)

        # upload the model and centroid to the device again
        self.centroids.to(device)
        self.autoencoder.to(device)
        self.to(device)
        print(f"Set {len(self.centroids)} means: {self.centroids}")

    @staticmethod
    def set_random_centroids(num_centroids, size_centroids):
        centroids = nn.Parameter(torch.rand([num_centroids, size_centroids]))
        nn.init.xavier_uniform_(centroids)
        return centroids

    @staticmethod
    def set_kmeans_centroids(encoder: nn.Module, inputs, true_labels, num_centroids):
        with torch.no_grad():
            embeddings = encoder(inputs)

            # initialize centroids with the highest purity
            initial_centroids, best_metric = None, None
            for i in range(N_INITIAL_TRIALS):
                method = KMeans(n_clusters=num_centroids, n_init=1)
                predictions = method.fit_predict(embeddings.numpy())
                predictions = torch.Tensor(predictions)
                initial_metric = purity_score(true_labels, predictions)

                if best_metric is None or initial_metric > best_metric:
                    best_metric = initial_metric
                    initial_centroids = method.cluster_centers_

        print(f"Best initial purity: {best_metric}")
        return nn.Parameter(torch.Tensor(initial_centroids))
