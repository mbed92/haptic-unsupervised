import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SlipDataset(Dataset):

    def __init__(self, path, label_key, standarize=True):
        # load CSV from path
        data = pd.read_csv(path, index_col=[0])

        # pick columns by names
        self.signals = data.drop(label_key, axis=1).to_numpy()
        self.labels = data[label_key].to_numpy()
        self.mean, self.std = np.mean(self.signals, 0, keepdims=True), np.std(self.signals, 0, keepdims=True)
        self.p_target = None

        if standarize:
            self._standarize()

    def _standarize(self):
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        if self.p_target is None:
            return self.signals[index], self.labels[index]
        return self.signals[index], self.labels[index], self.p_target[index]

    def __add__(self, other_slip):
        self.mean = (self.mean + other_slip.mean) / 2.0
        self.std = (self.std + other_slip.std) / 2.0
        self.signals = np.concatenate([self.signals, other_slip.signals], 0)
        self.labels = np.concatenate([self.labels, other_slip.labels], 0)
        return self
