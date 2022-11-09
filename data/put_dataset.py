import pickle

import numpy as np
from torch.utils.data import Dataset


class HapticDataset(Dataset):
    def __init__(self, path, key, signal_start, signal_length, standarize=True):
        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            self.signals = np.asarray([v["signal"] for v in pickled[key]])
            self.signals = np.transpose(self.signals, [0, 2, 1])
            self.labels = np.asarray([v["label"] for v in pickled[key]])
            self.mean, self.std = pickled['signal_stats']
            self.mean = self.mean[:, np.newaxis]
            self.std = self.std[:, np.newaxis]

        self.num_classes = len(np.unique(self.labels))
        self.signal_start = signal_start
        self.signal_length = signal_length

        if standarize:
            self._standarize()

    def _standarize(self):
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]

    def __add__(self, other_haptic):
        self.mean = (self.mean + other_haptic.mean) / 2.0
        self.std = (self.std + other_haptic.std) / 2.0
        self.signals = np.concatenate([self.signals, other_haptic.signals], 0)
        self.labels = np.concatenate([self.labels, other_haptic.labels], 0)
        self.num_classes = len(np.unique(self.labels))
        return self
