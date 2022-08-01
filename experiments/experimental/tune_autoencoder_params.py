from functools import partial

import torch
import torch.nn as nn
import yaml
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

import submodules.haptic_transformer.utils as utils_haptr
import utils
from data import helpers
from models.autoencoders import TimeSeriesAutoencoderConfig, TimeSeriesAutoencoder
from train_stacked_autoencoder import train_epoch, test_epoch

torch.manual_seed(42)


def tune_train(config, const_config, train_dataloader, test_dataloader):
    nn_params = TimeSeriesAutoencoderConfig()
    nn_params.data_shape = const_config["data_shape"]
    nn_params.stride = const_config["stride"]
    nn_params.kernel = config["kernel"]
    nn_params.activation = config["activation"]
    nn_params.dropout = config["dropout"]
    nn_params.num_heads = const_config["num_heads"]
    nn_params.use_attention = config["use_attention"]
    autoencoder = TimeSeriesAutoencoder(nn_params)
    device = utils.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # train the autoencoder
    backprop_config_ae = utils.ops.BackpropConfig()
    backprop_config_ae.model = autoencoder
    backprop_config_ae.lr = config["lr"]
    backprop_config_ae.eta_min = const_config["eta_min"]
    backprop_config_ae.epochs = const_config["epochs"]
    backprop_config_ae.weight_decay = config["weight_decay"]
    backprop_config_ae.optimizer = const_config["optimizer"]

    # train & test
    optimizer, scheduler = utils.ops.backprop_init(backprop_config_ae)
    for epoch in range(const_config["epochs"]):
        train_loss = train_epoch(autoencoder, train_dataloader, optimizer, device)
        scheduler.step()
        test_loss, _ = test_epoch(autoencoder, test_dataloader, device)
        tune.report(train_loss=train_loss.get(), test_loss=test_loss.get())
    print("Finished training")


if __name__ == '__main__':
    with open("../../config/put.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_ds, _, test_ds = helpers.load_dataset(config)

    # prepare configuration files
    tune_config = {
        "kernel": tune.choice([3, 5, 7, 9, 11]),
        "activation": tune.choice([nn.ReLU(), nn.GELU(), nn.ELU(), nn.LeakyReLU(), nn.SiLU()]),
        "dropout": tune.loguniform(0.1, 0.5),
        "lr": tune.loguniform(5e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-4, 1e-3),
        "use_attention": tune.choice([True, False])
    }

    const_config = {
        "batch_size": 256,
        "stride": 2,
        "num_heads": 1,
        "log_dir": utils_haptr.log.logdir_name('./', 'tune'),
        "epochs": 1000,
        "eta_min": 1e-4,
        "num_samples": 10,
        "data_shape": (train_ds.signal_length, train_ds.mean.shape[-1]),
        "optimizer": torch.optim.AdamW
    }

    # load dataset
    train_dataloader = DataLoader(train_ds, batch_size=const_config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=const_config["batch_size"], shuffle=True)

    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=const_config["epochs"],
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["train_loss", "test_loss"])

    result = tune.run(
        partial(tune_train,
                const_config=const_config, train_dataloader=train_dataloader, test_dataloader=test_dataloader),
        resources_per_trial={"cpu": 5, "gpu": 1},
        config=tune_config,
        num_samples=const_config["num_samples"],
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("test_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final test loss: {}".format(best_trial.last_result["test_loss"]))
