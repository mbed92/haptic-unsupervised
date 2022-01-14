import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset


class TouchingDataset(Dataset):
    def __init__(self, path, directions, signal_start=90, signal_length=90, standarize=True):
        pickled = loadmat(path)

        self.signals, self.labels = list(), list()
        for key in pickled.keys():
            for direction in directions:
                if direction in key:
                    if 'steps_' in key:
                        self.signals.append(pickled[key][:, signal_start:signal_start + signal_length, :])
                    elif 'labels_' in key:
                        self.labels.append(pickled[key][0])
                else:
                    continue

        if len(self.signals) > 0 and len(self.labels) > 0:
            self.signals = np.concatenate(self.signals)
            self.labels = np.concatenate(self.labels)
            assert self.labels.shape[0] == self.signals.shape[0]
        else:
            raise ValueError("Empty dataset.")

        self.num_classes = 11
        self.mean, self.std = np.mean(self.signals, (0, 1), keepdims=True), np.std(self.signals, (0, 1), keepdims=True)
        self.weights = np.ones(self.num_classes)
        self.signal_start = signal_start
        self.signal_length = signal_length

        if standarize:
            self._standarize()

    def _standarize(self):
        if self.signals is not None:
            self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        if type(self.signals) is np.ndarray:
            return self.signals.shape[0]
        return 0

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]

    def __add__(self, other_touching):
        self.mean = (self.mean + other_touching.mean) / 2.0
        self.std = (self.std + other_touching.std) / 2.0
        self.weights = (self.weights + other_touching.weights) / 2.0
        self.signals = np.concatenate([self.signals, other_touching.signals], 0)
        self.labels = np.concatenate([self.labels, other_touching.labels], 0)
        return self