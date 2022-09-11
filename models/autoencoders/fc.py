""" Fully connected AutEncoder """

import torch
import torch.nn as nn


class FullyConnectedAutoencoderConfig:
    data_shape: iter
    dropout: float
    activation: nn.Module
    latent_size: int


class FullyConnectedAutoencoder(nn.Module):

    def __init__(self, cfg: FullyConnectedAutoencoderConfig):
        super().__init__()
        self.encoder_layers = nn.Sequential(
            nn.Linear(cfg.data_shape[0], cfg.latent_size),
            nn.BatchNorm1d(cfg.latent_size),
            nn.Dropout(cfg.dropout),
            cfg.activation,
            nn.Linear(cfg.latent_size, cfg.latent_size)
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(cfg.latent_size, cfg.data_shape[0]),
            nn.BatchNorm1d(cfg.data_shape[0]),
            nn.Dropout(cfg.dropout),
            cfg.activation,
            nn.Linear(cfg.data_shape[0], cfg.data_shape[0])
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
