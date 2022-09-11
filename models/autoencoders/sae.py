""" Stacked AutEncoder """

from math import ceil

import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel: int = 3,
                 stride: int = 2,
                 act_enc: nn.Module = nn.ReLU,
                 act_dec: nn.Module = nn.ReLU,
                 dropout: float = 0.0,
                 **kwargs):

        super().__init__()
        padding = ceil(kernel / 2) - 1

        # build the encoder
        encoder_layers = list() if "first_layer" not in kwargs.keys() else [kwargs["first_layer"]]
        encoder_layers.append(nn.Conv1d(c_in, c_out, kernel, stride, padding=padding))
        if act_enc is not None:
            encoder_layers.append(nn.BatchNorm1d(c_out))
            encoder_layers.append(act_enc)
            encoder_layers.append(nn.Dropout(dropout))

        # build the decoder
        decoder_layers = list()
        decoder_layers.append(nn.ConvTranspose1d(c_out, c_in, kernel, stride,
                                                 padding=padding, output_padding=stride - 1))
        if act_dec is not None:
            decoder_layers.append(nn.BatchNorm1d(c_in))
            decoder_layers.append(act_dec)
            decoder_layers.append(nn.Dropout(dropout))

        if "last_layer" in kwargs.keys():
            decoder_layers.append(kwargs["last_layer"])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


class TimeSeriesAutoencoderConfig:
    data_shape: iter
    stride: int
    kernel: int
    dropout: float
    activation: nn.Module


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesAutoencoderConfig):
        super().__init__()
        assert len(cfg.data_shape) == 2

        # the SAE1 with additional mapping between BxNx1 <-> BxNxK
        self.sae1 = SAE(16, 32, cfg.kernel, cfg.stride, cfg.activation, cfg.activation, cfg.dropout,
                        first_layer=nn.Conv1d(cfg.data_shape[-1], 16, 1, 1),
                        last_layer=nn.Conv1d(16, cfg.data_shape[-1], 1, 1))
        self.sae2 = SAE(32, 64, cfg.kernel, cfg.stride, cfg.activation, cfg.activation, cfg.dropout)
        self.sae3 = SAE(64, 128, cfg.kernel, cfg.stride, cfg.activation, cfg.activation, cfg.dropout)
        self.sae4 = SAE(128, 1, cfg.kernel, cfg.stride, cfg.activation, cfg.activation, cfg.dropout)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)
        x = torch.squeeze(x, 1)
        return x

    def decoder(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        x = self.sae1.decoder(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
