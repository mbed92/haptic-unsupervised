import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset


class TouchingDataset(Dataset):
    def __init__(self, path, directions, classes, signal_start=90, signal_length=90, standarize=True, reorder=True):
        self.num_classes = 11
        pickled = loadmat(path)
        self.signals, self.labels = list(), list()
        for key in pickled.keys():
            for direction in directions:
                if direction in key:
                    if 'steps_' in key:
                        self.signals.append(pickled[key][:, signal_start:signal_start + signal_length, :])
                    elif 'labels_' in key:
                        self.labels.append(pickled[key][0] - 1)  # classes are numbered from 1 to 11
                else:
                    continue

        if len(self.signals) > 0 and len(self.labels) > 0:
            self.signals = np.concatenate(self.signals, 0)
            self.labels = np.concatenate(self.labels, 0)
            assert self.labels.shape[0] == self.signals.shape[0]
        else:
            raise ValueError("Empty dataset.")

        # pick only chosen classes if specified
        if type(classes) is list and 0 < len(classes):
            idx = np.argwhere([self.labels == c for c in classes])
            self.labels = self.labels[idx[:, -1]]
            self.signals = self.signals[idx[:, -1]]

            if reorder:
                for new_class_idx, c in enumerate(classes):
                    idx = np.argwhere(self.labels == c)
                    self.labels[idx[:, -1]] = new_class_idx

                self.num_classes = len(classes)

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
        return self.signals[index].T, self.labels[index]

    def __add__(self, other_touching):
        self.mean = (self.mean + other_touching.mean) / 2.0
        self.std = (self.std + other_touching.std) / 2.0
        self.weights = (self.weights + other_touching.weights) / 2.0
        self.signals = np.concatenate([self.signals, other_touching.signals], 0)
        self.labels = np.concatenate([self.labels, other_touching.labels], 0)
        return self
