import os

from .biotac_dataset import BiotacDataset
from .mock_dataset import MockDataset
from .put_dataset import HapticDataset
from .slip_dataset import SlipDataset
from .touching_dataset import TouchingDataset


def load_samples_to_device(data, device):
    s = data[0].to(device).float()
    labels = data[-1].to(device)
    return s, labels


def load_dataset(config):
    ds_type = config["dataset_type"].lower()

    if ds_type == "put":
        data_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = HapticDataset(data_path,
                                 key="train_ds",
                                 signal_start=config['signal_start'],
                                 signal_length=config['signal_length'],
                                 standarize=False)
        val_ds = HapticDataset(data_path, key="val_ds",
                               signal_start=config['signal_start'],
                               signal_length=config['signal_length'],
                               standarize=False)
        test_ds = HapticDataset(data_path, key="test_ds",
                                signal_start=config['signal_start'],
                                signal_length=config['signal_length'],
                                standarize=False)
        total_dataset = train_ds + val_ds + test_ds
        total_dataset._standarize()

    elif ds_type == "touching":
        data_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        picked_classes = config['train_val_classes'] if 'train_val_classes' in config.keys() else []
        train_ds = TouchingDataset(data_path,
                                   directions=config['train_val_directions'],
                                   classes=picked_classes,
                                   signal_start=config['signal_start'],
                                   signal_length=config['signal_length'], standarize=False)

        picked_classes = config['test_classes'] if 'test_classes' in config.keys() else []
        test_ds = TouchingDataset(data_path,
                                  directions=config['test_directions'],
                                  classes=picked_classes,
                                  signal_start=config['signal_start'],
                                  signal_length=config['signal_length'], standarize=False)
        total_dataset = train_ds + test_ds
        total_dataset._standarize()

    elif ds_type == "biotac2":
        data_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = BiotacDataset(data_path, "train", standarize=False, label_name=config['label_name'])
        test_ds = BiotacDataset(data_path, "test", standarize=False, label_name=config['label_name'])
        total_dataset = train_ds + test_ds
        total_dataset._standarize()

    elif ds_type == "slip":
        data_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        total_dataset = SlipDataset(data_path, label_key=config['label_key'])

    elif ds_type == "mock":
        total_dataset = MockDataset()

    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: put, touching, biotac2, mock")

    return total_dataset
