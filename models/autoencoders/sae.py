""" Stacked AutEncoder """

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


class TimeSeriesAutoencoderConfig:
    data_shape: iter
    use_attention: bool
    stride: int
    kernel: int
    dropout: float
    num_heads: int
    activation: nn.Module
    embedding_size: int


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesAutoencoderConfig):
        super().__init__()
        assert len(cfg.data_shape) == 2

        sig_length, num_channels = cfg.data_shape
        self.sae1 = SAE(num_channels, 32, cfg.kernel, cfg.stride, cfg.activation, None)  # reconstructs
        self.sae2 = SAE(32, 64, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae3 = SAE(64, 128, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae4 = SAE(128, 3, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]

        last_layer_size = int(sig_length / (cfg.stride ** len(self.sae_modules)))
        self.attention_layer = nn.MultiheadAttention(embed_dim=last_layer_size, num_heads=1,
                                                     dropout=0.2) if cfg.use_attention else None

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)

        if self.attention_layer is not None:
            x_attn, _ = self.attention_layer(x, x, x)
            x = x + x_attn

        x = x.reshape((x.shape[0], -1))
        return x

    def decoder(self, x):
        x = x.reshape((x.shape[0], 3, -1))
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        x = self.sae1.decoder(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
