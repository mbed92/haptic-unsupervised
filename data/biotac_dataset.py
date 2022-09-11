import pickle

import numpy as np
from torch.utils.data import Dataset

SIGNAL_START = 2
SIGNAL_END = 74
NUM_CLASSES_TRAIN = 41
NUM_CLASSES_TEST = 10
CLASS_NAME_IDX = 0
OUTCOME_IDX = 1
PALM_ORIENTATION_IDX = -1


class BiotacDataset(Dataset):

    def __init__(self, path, key, standarize=True):
        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            self.signals = pickled[key][:, SIGNAL_START:SIGNAL_END]
            self.labels = pickled[key][:, CLASS_NAME_IDX]
            self.meta = pickled[key][:, [OUTCOME_IDX, PALM_ORIENTATION_IDX]]

        self.num_classes = NUM_CLASSES_TRAIN if key == "train" else NUM_CLASSES_TEST
        self.mean, self.std = np.mean(self.signals, 0, keepdims=True), np.std(self.signals, 0, keepdims=True)
        self.weights = np.ones(self.num_classes)

        if standarize:
            self._standarize()

    def _standarize(self):
        self.signals = (self.signals - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        sig = self.signals[index]['label']
        label = self.signals[index]['label']
        return sig, label

    def __add__(self, other_biotac):
        self.mean = (self.mean + other_biotac.mean) / 2.0
        self.std = (self.std + other_biotac.std) / 2.0
        self.signals = np.concatenate([self.signals, other_biotac.signals], 0)
        self.labels = np.concatenate([self.labels, other_biotac.labels], 0)
        self.weights = np.concatenate([self.weights, other_biotac.weights], 0) / 2.0
        return self
