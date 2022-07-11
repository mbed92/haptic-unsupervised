import torch.nn as nn


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
