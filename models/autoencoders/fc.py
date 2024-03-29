""" Fully connected AutEncoder with the attention variant """
from argparse import Namespace

import torch
import torch.nn as nn

from models import autoencoders


class FullyConnectedAutoencoderConfig:
    data_shape: iter
    activation: nn.Module
    latent_size: int


class FullyConnectedAutoencoder(nn.Module):

    def __init__(self, cfg: FullyConnectedAutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder_layers = nn.Sequential(
            nn.Linear(cfg.data_shape[0], 128),
            nn.BatchNorm1d(128),
            cfg.activation,
            nn.Linear(128, cfg.latent_size),
            nn.BatchNorm1d(cfg.latent_size),
            cfg.activation,
            nn.Linear(cfg.latent_size, cfg.latent_size)
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(cfg.latent_size, cfg.latent_size),
            nn.BatchNorm1d(cfg.latent_size),
            cfg.activation,
            nn.Linear(cfg.latent_size, 128),
            nn.BatchNorm1d(128),
            cfg.activation,
            nn.Linear(128, cfg.data_shape[0])
        )

    def encoder(self, inputs):
        z = self.encoder_layers(inputs)
        return torch.squeeze(z, 1)

    def decoder(self, inputs):
        x = self.decoder_layers(inputs)
        return x

    def forward(self, inputs):
        z = self.encoder(inputs)
        x = self.decoder(z)
        return x

    def setup(self, args: Namespace):
        backprop_config = autoencoders.ops.BackpropConfig()
        backprop_config.model = self
        backprop_config.optimizer = torch.optim.AdamW
        backprop_config.lr = args.lr
        backprop_config.eta_min = args.eta_min
        backprop_config.epochs = args.epochs_ae
        backprop_config.weight_decay = args.weight_decay
        return autoencoders.ops.backprop_init(backprop_config)
