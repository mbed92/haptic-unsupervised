''' Based on Unsupervised Deep Embedding for Clustering Analysis '''

import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.GELU()
        )
        self.encoder_drop = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(out_size, in_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder_drop = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.encoder(self.encoder_drop(inputs))
        return self.decoder(self.decoder_drop(x))

    def set_dropout(self, rate):
        assert 0.0 <= rate < 1.0
        self.encoder_drop.p = rate
        self.decoder_drop.p = rate


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, data_shape: list, embedding_size: int):
        super().__init__()
        assert len(data_shape) > 0

        flatten_data_size = 1
        for ds in data_shape:
            flatten_data_size *= ds
        flatten_data_size = int(flatten_data_size)

        self._input_data_size = data_shape
        self.sae1 = SAE(flatten_data_size, 500)
        self.sae2 = SAE(500, 500)
        self.sae3 = SAE(500, 2000)
        self.sae4 = SAE(2000, embedding_size)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            self.sae1.encoder,
            self.sae2.encoder,
            self.sae3.encoder,
            self.sae4.encoder
        )

        self.decoder = nn.Sequential(
            self.sae4.decoder,
            self.sae3.decoder,
            self.sae2.decoder,
            self.sae1.decoder,
            Reshape(self._input_data_size)
        )

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
