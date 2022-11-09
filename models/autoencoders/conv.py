""" Stacked AutEncoder """
from math import ceil

import torch
import torch.nn as nn


class TimeSeriesConvAutoencoderConfig:
    data_shape: iter
    kernel: int
    activation: nn.Module


def change_channels(in_chan, out_chan):
    return nn.Conv1d(in_chan, out_chan, kernel_size=1)


def encoder_block(in_f, out_f, kernel, stride, padding, activation):
    return nn.Sequential(
        nn.Conv1d(in_f, out_f, kernel, stride, padding),
        nn.BatchNorm1d(out_f),
        activation
    )


def decoder_block(in_f, out_f, kernel, stride, padding, activation):
    return nn.Sequential(
        nn.Conv1d(in_f, out_f, kernel, stride, padding),
        nn.BatchNorm1d(out_f),
        activation,
        nn.Upsample(scale_factor=2.0, mode='linear', align_corners=False)
    )


class TimeSeriesConvAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesConvAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        assert len(cfg.data_shape) == 2

        padding = ceil(cfg.kernel / 2) - 1
        self.encoder_layers = nn.Sequential(
            change_channels(cfg.data_shape[0], 128),
            encoder_block(128, 128, cfg.kernel, 2, padding, cfg.activation),
            encoder_block(128, 64, cfg.kernel, 2, padding, cfg.activation),
            encoder_block(64, 64, cfg.kernel, 2, padding, cfg.activation),
            change_channels(64, 1)
        )

        self.decoder_layers = nn.Sequential(
            change_channels(1, 64),
            decoder_block(64, 64, cfg.kernel, 1, padding, cfg.activation),
            decoder_block(64, 128, cfg.kernel, 1, padding, cfg.activation),
            decoder_block(128, 128, cfg.kernel, 1, padding, cfg.activation),
            change_channels(128, cfg.data_shape[0])
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
