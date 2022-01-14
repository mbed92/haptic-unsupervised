import os

from data import TouchingDataset


def load_dataset(config):
    if config["dataset_type"].lower() in ["put", "qcat"]:
        from submodules.haptic_transformer.utils.dataset import load_dataset as load_dataset_haptr
        train_ds, val_ds, test_ds = load_dataset_haptr(config)
    elif config["dataset_type"].lower() == "touching":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = TouchingDataset(dataset_path,
                                   directions=config['train_val_directions'],
                                   signal_start=config['signal_start'],
                                   signal_length=config['signal_length'])

        val_ds = None

        test_ds = TouchingDataset(dataset_path,
                                  directions=config['train_val_directions'],
                                  signal_start=config['signal_start'],
                                  signal_length=config['signal_length'])
    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: QCAT, PUT, TOUCHING")

    return train_ds, val_ds, test_ds
