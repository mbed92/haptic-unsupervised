import torch.nn as nn

from models.sae import SAE


class RCNNAutoencoderConfig:
    input_num_channels: int
    sequence_length: int
    latent_dim: int
    num_layers: int
    kernel: int
    dropout: float
    hidden_units: int


class RCNNAutoencoder(nn.Module):
    def __init__(self, cfg: RCNNAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.GRU(cfg.input_num_channels, cfg.latent_dim, dropout=cfg.dropout, num_layers=cfg.num_layers)

        self.linear = nn.Sequential(
            nn.Linear(self.cfg.latent_dim * self.cfg.num_layers, cfg.hidden_units),
            nn.BatchNorm1d(cfg.hidden_units),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_units, cfg.hidden_units),
            nn.BatchNorm1d(cfg.hidden_units),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_units, self.cfg.sequence_length * cfg.input_num_channels)
        )

    def encoder(self, inputs):
        batch_size, num_features, seq_len = inputs.shape
        x = inputs.reshape((seq_len, batch_size, num_features))
        _, x = self.rnn(x)
        return x.reshape((batch_size, self.cfg.latent_dim * self.cfg.num_layers))

    def decoder(self, latent_vectors):
        y = self.linear(latent_vectors)
        y = y.reshape(y.shape[0], self.cfg.input_num_channels, self.cfg.sequence_length)
        return y

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
        self.sae1 = SAE(num_channels, 32, cfg.kernel, cfg.stride, cfg.activation, None)  # reconstructs
        self.sae2 = SAE(32, 64, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae3 = SAE(64, 128, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae4 = SAE(128, 16, cfg.kernel, cfg.stride, cfg.activation, cfg.activation)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]

        embedding_size = int(sig_length / (cfg.stride ** len(self.sae_modules)))
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=1,
                                                     dropout=0.2) if cfg.use_attention else None

        self.downsample = nn.Linear(16 * embedding_size, embedding_size)
        self.upsample = nn.Linear(embedding_size, 16 * embedding_size)

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)

        if self.attention_layer is not None:
            x_attn, _ = self.attention_layer(x, x, x)
            x = x + x_attn

        x = self.downsample(x.reshape((x.shape[0], -1)))
        return x

    def decoder(self, x):
        x = self.upsample(x).reshape((x.shape[0], 16, -1))
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        x = self.sae1.decoder(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
