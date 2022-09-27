import pickle

import numpy as np
from torch.utils.data import Dataset

SIGNAL_START = 2
SIGNAL_END = 74
NUM_CLASSES_TRAIN = 41
NUM_CLASSES_TEST = 10
CLASS_NAME_IDX = 0
CLASS_ID_IDX = 75
OUTCOME_IDX = 1
PALM_ORIENTATION_IDX = -1


class BiotacDataset(Dataset):

    def __init__(self, path, key, standarize=True):
        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            self.signals = pickled[key][:, SIGNAL_START:SIGNAL_END].astype(np.float32)
            self.labels = pickled[key][:, CLASS_ID_IDX].astype(np.int32)
            self.meta = pickled[key][:, [CLASS_NAME_IDX, OUTCOME_IDX, PALM_ORIENTATION_IDX]]

        self.num_classes = len(np.unique(self.labels))
        self.mean, self.std = np.mean(self.signals, 0, keepdims=True), np.std(self.signals, 0, keepdims=True)
        self.weights = np.ones(self.num_classes)

        if standarize:
            self._standarize()

    def _standarize(self):
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]

    def __add__(self, other_biotac):
        self.mean = (self.mean + other_biotac.mean) / 2.0
        self.std = (self.std + other_biotac.std) / 2.0
        self.signals = np.concatenate([self.signals, other_biotac.signals], 0)
        self.labels = np.concatenate([self.labels, other_biotac.labels], 0)
        self.weights = np.concatenate([self.weights, other_biotac.weights], 0)
        self.meta = np.concatenate([self.meta, other_biotac.meta], 0)
        self.num_classes = self.num_classes + other_biotac.num_classes
        return self
