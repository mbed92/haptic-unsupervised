import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

import utils
from models import TimeSeriesAutoencoder

N_INITIAL_TRIALS = 30


class ClusteringModel(nn.Module):

    def __init__(self, data_shape: list, num_clusters: int, device, alpha=10.0):
        super().__init__()

        self.num_clusters = num_clusters
        self.alpha = alpha
        self.autoencoder = TimeSeriesAutoencoder(data_shape)
        self.centroids = nn.Parameter(torch.rand([self.num_clusters, 10]).to(device))

    def forward(self, inputs):
        x = self.autoencoder.encoder(inputs)
        q = self.t_student(x)
        y = self.autoencoder.decoder(x)

        return {
            'assignments': q,
            'reconstruction': y,
        }

    def t_student(self, x):
        # dist = torch.linalg.norm(x[:, None] - self.centroids, dim=-1)
        dist = torch.abs(F.cosine_similarity(x[:, None], self.centroids, dim=-1))
        q = 1.0 / (1.0 + dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        a = torch.sum(q, -1, keepdim=True)
        return q / a

    def predict_soft_assignments(self, data):
        return self.t_student(self.autoencoder.encoder(data))

    def predict_class(self, data):
        return torch.argmax(self.predict_soft_assignments(data), -1)

    def from_pretrained(self, model_path, dataloader, device):
        # centroids initialization
        self.autoencoder = torch.load(model_path)
        self.autoencoder.to(device)

        # remove regularization (dropout and batchnorm)
        for m in self.autoencoder.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Dropout):
                m.eval()

        # find latent vectors
        with torch.no_grad():
            x, y = utils.dataset.get_total_data_from_dataloader(dataloader)
            x_emb = self.autoencoder.encoder(x.permute(0, 2, 1).to(device))

            # taking the best centroid according to the accuracy
            initial_centroids, best_accuracy = None, None

            for i in range(N_INITIAL_TRIALS):
                kmeans = KMeans(n_clusters=self.num_clusters, n_init=1)
                predictions = torch.Tensor(kmeans.fit_predict(x_emb.cpu().numpy()))
                initial_accuracy = utils.clustering.clustering_accuracy(y.to(predictions.device), predictions).numpy()

                if best_accuracy is None or initial_accuracy > best_accuracy:
                    best_accuracy = initial_accuracy
                    initial_centroids = kmeans.cluster_centers_

        # save the best centroids for optimization
        self.centroids.data = torch.Tensor(initial_centroids).type(torch.float32).to(device)


def distribution_hardening(q):
    p = torch.div(q ** 2, torch.sum(q, 0, keepdim=True))
    return torch.div(p, torch.sum(p, 1, keepdim=True))
