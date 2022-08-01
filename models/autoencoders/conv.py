""" Stacked AutEncoder """
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


class TimeSeriesBinaryAutoencoderConfig:
    data_shape: iter
    stride: int
    kernel: int
    dropout: float
    activation: nn.Module
    embedding_size: int


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


class TimeSeriesBinaryAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesBinaryAutoencoderConfig):
        super().__init__()
        assert len(cfg.data_shape) == 2
        self.embedding_size = 10

        padding = ceil(cfg.kernel / 2) - 1
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(cfg.data_shape[-1], 64, 1, 1),
            encoder_block(64, 32, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            encoder_block(32, 64, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            encoder_block(64, 128, cfg.kernel, 2, padding, cfg.activation, cfg.dropout),
            nn.Conv1d(128, self.embedding_size, 1, 1)
        )

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self.embedding_size, 128, 1, 1),
            decoder_block(128, 64, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            decoder_block(64, 32, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            decoder_block(32, 16, cfg.kernel, 1, padding, cfg.activation, cfg.dropout),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv1d(16, cfg.data_shape[-1], 1, 1)
        )

    def encoder(self, inputs):
        z = self.encoder_layers(inputs)
        # z = torch.argmax(z_logits, dim=1)
        # z = F.one_hot(z, num_classes=self.vocabulary_size).permute(0, 2, 1).float()
        return z

    def forward(self, inputs):
        z = self.encoder(inputs)
        x_reconstructed = self.decoder_layers(z)
        return x_reconstructed
