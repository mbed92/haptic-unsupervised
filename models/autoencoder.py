''' Based on Unsupervised Deep Embedding for Clustering Analysis '''

import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, in_size, out_size, activation_encoder: nn.Module, activation_decoder: nn.Module, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('lin', nn.Linear(in_size, out_size))
        if activation_encoder is not None:
            self.encoder.add_module("activation", activation_encoder)
        self.encoder_drop = nn.Dropout(dropout)

        self.decoder = nn.Sequential()
        self.decoder.add_module('lin', nn.Linear(out_size, in_size))
        if activation_decoder is not None:
            self.decoder.add_module("activation", activation_decoder)
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
        self.sae1 = SAE(flatten_data_size, 500, nn.GELU(), None)
        self.sae2 = SAE(500, 500, nn.GELU(), nn.GELU())
        self.sae3 = SAE(500, 2000, nn.GELU(), nn.GELU())
        self.sae4 = SAE(2000, embedding_size, None, nn.GELU())

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
            self.sae1.decoder
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
