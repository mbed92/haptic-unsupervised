import copy
import os

import torch
from torch.utils.data import DataLoader

from .biotac_dataset import BiotacDataset
from .put_dataset import HapticDataset
from .touching_dataset import TouchingDataset


def get_total_data_from_dataloader(dataloader: DataLoader):
    x_list, y_list = list(), list()

    for sample in dataloader.dataset:
        x_list.append(torch.FloatTensor(sample[0]))
        y_list.append(sample[1])

    return torch.stack(x_list, 0).float().detach(), torch.FloatTensor(y_list).detach()


def load_samples_to_device(data, device):
    s = data[0].to(device).float()
    labels = data[-1].to(device)
    return s, labels


def prepare(old_ds: HapticDataset):
    new_ds = copy.deepcopy(old_ds)
    new_ds.signals = [d["signal"] for d in old_ds.signals]
    new_ds.labels = [d["label"] for d in old_ds.signals]
    return new_ds


def load_dataset(config):
    ds_type = config["dataset_type"].lower()

    if ds_type in ["put"]:
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = HapticDataset(dataset_path,
                                 key="train_ds",
                                 signal_start=config['signal_start'],
                                 signal_length=config['signal_length'], standarize=False)
        val_ds = HapticDataset(dataset_path, key="val_ds",
                               signal_start=config['signal_start'],
                               signal_length=config['signal_length'], standarize=False)
        test_ds = HapticDataset(dataset_path, key="test_ds",
                                signal_start=config['signal_start'],
                                signal_length=config['signal_length'], standarize=False)
        total_dataset = train_ds + val_ds + test_ds
        total_dataset._standarize()

    elif ds_type == "touching":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        picked_classes = config['train_val_classes'] if 'train_val_classes' in config.keys() else []
        train_ds = TouchingDataset(dataset_path,
                                   directions=config['train_val_directions'],
                                   classes=picked_classes,
                                   signal_start=config['signal_start'],
                                   signal_length=config['signal_length'], standarize=False)

        picked_classes = config['test_classes'] if 'test_classes' in config.keys() else []
        test_ds = TouchingDataset(dataset_path,
                                  directions=config['test_directions'],
                                  classes=picked_classes,
                                  signal_start=config['signal_start'],
                                  signal_length=config['signal_length'], standarize=False)
        total_dataset = train_ds + test_ds
        total_dataset._standarize()

    elif ds_type == "biotac2":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = BiotacDataset(dataset_path, "train", standarize=False)
        test_ds = BiotacDataset(dataset_path, "test", standarize=False)
        total_dataset = train_ds + test_ds
        total_dataset._standarize()

    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: QCAT, PUT, TOUCHING")

    return total_dataset
