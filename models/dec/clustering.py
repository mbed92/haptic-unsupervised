import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from utils.metrics import clustering_accuracy

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

    def from_pretrained(self, model_path: str, input_samples: torch.Tensor, true_labels: torch.Tensor,
                        device: torch.device, best_centroids: bool = True):
        # centroids initialization
        self.autoencoder = torch.load(model_path).cpu()

        # # remove regularization (dropout and batchnorm)
        # for m in self.autoencoder.modules():
        #     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Dropout):
        #         m.eval()

        # fit the best centroids
        x = input_samples.permute(0, 2, 1).float()
        if best_centroids:
            with torch.no_grad():
                embeddings = self.autoencoder.encoder(x)

                # initialize centroids
                initial_centroids, best_accuracy = None, None
                for i in range(N_INITIAL_TRIALS):
                    kmeans = KMeans(n_clusters=self.num_clusters, n_init=1)
                    predictions = torch.Tensor(kmeans.fit_predict(embeddings.numpy()))
                    initial_accuracy = clustering_accuracy(true_labels, predictions).numpy()

                    if best_accuracy is None or initial_accuracy > best_accuracy:
                        best_accuracy = initial_accuracy
                        initial_centroids = kmeans.cluster_centers_

                self.centroids.data = torch.Tensor(initial_centroids)
            print(f"Best initial accuracy: {best_accuracy}")
        else:
            embeddings_size = self.autoencoder.encoder(x).shape[-1]
            self.centroids = nn.Parameter(torch.rand([self.num_clusters, embeddings_size]))
            nn.init.xavier_uniform_(self.centroids)

        # upload the model and centroid to the device again
        self.centroids.to(device)
        self.autoencoder.to(device)
        self.to(device)
