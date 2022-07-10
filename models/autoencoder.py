import math

import torch
import torch.nn as nn

from models.sae import SAE


class LSTMAutoencoderConfig:
    input_num_channels: int
    sequence_length: int
    latent_dim: int
    num_layers: int
    kernel: int
    dropout: float


class LSTMAutoencoder(nn.Module):
    def __init__(self, cfg: LSTMAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm_in = nn.LSTM(cfg.input_num_channels, cfg.latent_dim, dropout=cfg.dropout, num_layers=cfg.num_layers)
        self.lstm_out = nn.LSTM(cfg.latent_dim, cfg.input_num_channels, dropout=cfg.dropout, num_layers=cfg.num_layers)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(cfg.input_num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(cfg.input_num_channels, cfg.input_num_channels, kernel_size=cfg.kernel,
                      padding=math.ceil(cfg.kernel / 2) - 1),
            nn.Dropout(cfg.dropout)
        )

    def encoder(self, inputs):
        batch_size, num_features, seq_len = inputs.shape
        x = inputs.reshape((seq_len, batch_size, num_features))
        _, (last_hidden, _) = self.lstm_in(x)
        last_hidden = last_hidden.repeat((self.cfg.sequence_length, 1, 1))
        last_hidden = torch.mean(last_hidden, 0)
        return last_hidden

    def decoder(self, latent_vectors):
        x = torch.unsqueeze(latent_vectors, 0)
        x = x.repeat((self.cfg.sequence_length, 1, 1))
        y, _ = self.lstm_out(x)
        y = y.permute(1, 2, 0)
        y = self.conv(y)
        return y.reshape((latent_vectors.shape[0], self.cfg.input_num_channels, self.cfg.sequence_length))

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class TimeSeriesAutoencoderConfig:
    data_shape: iter
    use_attention: bool
    stride: int
    kernel: int
    dropout: float
    num_heads: int
    activation: nn.Module


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, cfg: TimeSeriesAutoencoderConfig):
        super().__init__()
        assert len(cfg.data_shape) == 2

        sig_length, num_channels = cfg.data_shape
        self.sae1 = SAE(num_channels, 64, cfg.kernel, cfg.stride, cfg.activation, None)  # reconstructs
        self.sae2 = SAE(64, 128, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae3 = SAE(128, 256, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae4 = SAE(256, 3, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]

        # self attention layer
        embedding_size = int(sig_length / (cfg.stride ** len(self.sae_modules)))
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=1,
                                                     dropout=0.2) if cfg.use_attention else None

        self.max_pool = nn.MaxPool2d((3, 1))
        self.up_sample = nn.Linear(1, 3)

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)

        if self.attention_layer is not None:
            x_attn, _ = self.attention_layer(x, x, x)
            x = x + x_attn

        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        return x

    def decoder(self, x):
        x = torch.unsqueeze(x, -1)
        x = self.up_sample(x)
        x = x.view(x.shape[0], 3, -1)
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        x = self.sae1.decoder(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
