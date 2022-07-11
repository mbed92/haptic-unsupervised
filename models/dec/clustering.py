import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

import utils

N_INITIAL_TRIALS = 30


class ClusteringModel(nn.Module):

    def __init__(self, num_clusters: int, device, alpha=95.0):
        super().__init__()

        self.num_clusters = num_clusters
        self.alpha = alpha
        self.autoencoder = None
        self.centroids = nn.Parameter(torch.rand([self.num_clusters, 10]).to(device))
        nn.init.xavier_uniform_(self.centroids)

    def forward(self, inputs):
        x = self.autoencoder.encoder(inputs)
        q = self.t_student(x)
        y = self.autoencoder.decoder(x)

        return {
            'assignments': q,
            'reconstruction': y,
        }

    def t_student(self, x):
        dist = x[:, None] - self.centroids[None, :]
        dist = torch.linalg.norm(dist, dim=-1)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        a = torch.sum(q, 1, keepdim=True)
        return q / a

    def predict_soft_assignments(self, data):
        return self.t_student(self.autoencoder.encoder(data))

    def predict_class(self, data):
        return torch.argmax(self.predict_soft_assignments(data), -1)

    def from_pretrained(self, model_path, dataloader: DataLoader, device, best_centroids=True):
        # centroids initialization
        self.autoencoder = torch.load(model_path)
        self.autoencoder.to(device)

        # freeze decoder
        for sae in self.autoencoder.sae_modules:
            sae.decoder.requires_grad = False

        # remove regularization (dropout and batchnorm)
        for m in self.autoencoder.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Dropout):
                m.eval()

        # fit the best centroids
        if best_centroids:
            with torch.no_grad():

                # find latent vectors
                x_emb, y = list(), list()
                for data in dataloader:
                    x = data[0].to(device).permute(0, 2, 1).float()
                    emb = self.autoencoder.encoder(x)
                    x_emb.append(emb)
                    y.append(data[1])

                # taking the best centroid according to the accuracy
                x_emb = torch.concat(x_emb, 0)
                y = torch.concat(y, 0)

                # initialize centroids
                initial_centroids, best_accuracy = None, None
                for i in range(N_INITIAL_TRIALS):
                    kmeans = KMeans(n_clusters=self.num_clusters, n_init=1)
                    predictions = torch.Tensor(kmeans.fit_predict(x_emb.cpu().numpy()))
                    initial_accuracy = utils.clustering.clustering_accuracy(y.to(predictions.device),
                                                                            predictions).numpy()

                    if best_accuracy is None or initial_accuracy > best_accuracy:
                        best_accuracy = initial_accuracy
                        initial_centroids = kmeans.cluster_centers_

                self.centroids.data = torch.Tensor(initial_centroids).to(device)

            # save the best centroids for optimization
            print(f"Best initial accuracy: {best_accuracy}")
        else:
            # verify the size of embeddings
            data = dataloader.dataset[0]
            x = torch.Tensor(data[0])[None, :]
            x = x.to(device).permute(0, 2, 1).float()
            emb_size = self.autoencoder.encoder(x).shape[-1]
            self.centroids = nn.Parameter(torch.rand([self.num_clusters, emb_size]).to(device))
            nn.init.xavier_uniform_(self.centroids)
