from math import ceil

import torch.nn as nn


class SAE(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel: int = 3,
                 stride: int = 2,
                 act_enc: nn.Module = nn.ReLU,
                 act_dec: nn.Module = nn.ReLU,
                 dropout: float = 0.0):

        super().__init__()
        self.encoder, self.encoder_drop = self.build_encoder(c_in, c_out, kernel, act_enc, stride, dropout)
        self.decoder, self.decoder_drop = self.build_decoder(c_out, c_in, kernel, act_dec, stride, dropout)

    @staticmethod
    def build_encoder(c_in, c_out, kernel, activation, stride, dropout):
        layers = list()
        padding = ceil(kernel / 2) - 1
        layers.append(nn.Conv1d(c_in, c_out, kernel, stride, padding=padding))

        if activation is not None:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation)

        return nn.Sequential(*layers), nn.Dropout(dropout)

    @staticmethod
    def build_decoder(c_in, c_out, kernel, activation, stride, dropout):
        layers = list()
        padding = ceil(kernel / 2) - 1
        layers.append(nn.ConvTranspose1d(c_in, c_out, kernel, stride,
                                         padding=padding, output_padding=stride - 1))

        if activation is not None:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation)

        return nn.Sequential(*layers), nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.encoder_drop(self.encoder(inputs))
        return self.decoder_drop(self.decoder(x))

    def set_dropout(self, rate):
        assert 0.0 <= rate < 1.0
        self.encoder_drop.p = rate
        self.decoder_drop.p = rate
