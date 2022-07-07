import os
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import TouchingDataset
from submodules.haptic_transformer.data import HapticDataset, QCATDataset
from submodules.haptic_transformer.utils.dataset import load_dataset as load_dataset_haptr


def get_total_data_from_dataloader(dataloader: DataLoader):
    x_list, y_list = list(), list()

    for sample in dataloader.dataset:
        x_list.append(torch.FloatTensor(sample[0]))
        y_list.append(sample[1])

    return torch.stack(x_list, 0).float(), torch.FloatTensor(y_list)


def pick_haptic_dataset(ds: HapticDataset, classes: list):
    labels = [sig['label'] for sig in ds.signals]
    chosen_idx = np.argwhere([bool(l in classes) for idx, l in enumerate(labels)]).flatten()
    new_signals = [ds.signals[i] for i in chosen_idx]
    new_ds = copy(ds)
    new_ds.signals = new_signals
    return new_ds


def pick_qcat_dataset(ds: QCATDataset, classes: list):
    raise NotImplementedError("Cannot filter the dataset.")


def load_samples_to_device(data, device):
    s = data[0].to(device).float()
    labels = data[-1].to(device)
    return s, labels


def load_dataset(config):
    if config["dataset_type"].lower() in ["put", "qcat"]:
        train_ds, val_ds, test_ds = load_dataset_haptr(config)
        train_ds += val_ds

        if 'test_classes' in config.keys() or 'train_val_classes' in config.keys():
            ds = train_ds + val_ds + test_ds
            val_ds = None

            if config["dataset_type"].lower() == "put":
                train_ds = pick_haptic_dataset(ds, config['train_val_classes'])
                test_ds = pick_haptic_dataset(ds, config['test_classes'])
            elif config["dataset_type"].lower() == "qcat":
                train_ds = pick_qcat_dataset(ds, config['train_val_classes'])
                test_ds = pick_qcat_dataset(ds, config['test_classes'])
            else:
                raise NotImplementedError("Cannot filter the dataset.")

    elif config["dataset_type"].lower() == "touching":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        picked_classes = config['train_val_classes'] if 'train_val_classes' in config.keys() else []
        train_ds = TouchingDataset(dataset_path,
                                   directions=config['train_val_directions'],
                                   classes=picked_classes,
                                   signal_start=config['signal_start'],
                                   signal_length=config['signal_length'],
                                   standarize=True)

        val_ds = None
        picked_classes = config['test_classes'] if 'test_classes' in config.keys() else []
        test_ds = TouchingDataset(dataset_path,
                                  directions=config['test_directions'],
                                  classes=picked_classes,
                                  signal_start=config['signal_start'],
                                  signal_length=config['signal_length'],
                                  standarize=False)
        test_ds.signals = (test_ds.signals - train_ds.mean) / train_ds.std
    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: QCAT, PUT, TOUCHING")

    return train_ds, val_ds, test_ds
