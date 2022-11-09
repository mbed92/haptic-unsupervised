import numpy as np
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset


class MockDataset(Dataset):

    def __init__(self, standarize=True):
        self.signals, self.labels = make_blobs(n_samples=5000, n_features=160, centers=3)
        self.signals = np.stack([self.signals] * 6).transpose(1, 0, 2)

        self.num_classes = len(np.unique(self.labels))
        self.mean, self.std = np.mean(self.signals, 0, keepdims=True), np.std(self.signals, 0, keepdims=True)

        if standarize:
            self._standarize()

    def _standarize(self):
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]

    def __add__(self, other_biotac):
        raise NotImplementedError("Dataset is only for testing purposes.")
