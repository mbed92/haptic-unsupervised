""" Stacked AutEncoder """
from math import ceil

import torch
import torch.nn as nn


class TimeSeriesConvAutoencoderConfig:
    data_shape: iter
    kernel: int
    dropout: float
    activation: nn.Module


def encoder_block(in_f, out_f, kernel, stride, padding, activation, dropout):
    return nn.Sequential(
        nn.Conv1d(in_f, out_f, kernel, stride, padding),
        activation,
        nn.BatchNorm1d(out_f),
        nn.Dropout(dropout)
    )


def decoder_block(in_f, out_f, kernel, stride, padding, activation, dropout):
    return nn.Sequential(
        nn.Conv1d(in_f, out_f, kernel, stride, padding),
        activation,
        nn.BatchNorm1d(out_f),
        nn.Dropout(dropout)
    )


class TimeSeriesConvAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesConvAutoencoderConfig):
        super().__init__()
        assert len(cfg.data_shape) == 2

        padding = ceil(cfg.kernel / 2) - 1
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(cfg.data_shape[0], 16, 1, 1),
            encoder_block(16, 32, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            encoder_block(32, 64, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            encoder_block(64, 128, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            nn.Conv1d(128, 1, 1, 1)
        )

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(1, 128, 1, 1),
            decoder_block(128, 64, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            decoder_block(64, 32, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            decoder_block(32, 16, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv1d(16, cfg.data_shape[0], 1, 1)
        )

    def encoder(self, inputs):
        z = self.encoder_layers(inputs)
        return torch.squeeze(z, 1)

    def decoder(self, inputs):
        inputs = torch.unsqueeze(inputs, 1)
        x = self.decoder_layers(inputs)
        return x

    def forward(self, inputs):
        z = self.encoder(inputs)
        x = self.decoder(z)
        return x
