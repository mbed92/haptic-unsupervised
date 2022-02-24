''' Based on Unsupervised Deep Embedding for Clustering Analysis '''
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, act_enc: nn.Module, act_dec: nn.Module, dropout: float):
        super().__init__()

        self.num_chan_in = c_in
        self.num_chan_out = c_out
        self.stride = stride
        self.dropout = dropout

        self.encoder, self.encoder_drop = self.build_encoder(c_in, c_out, act_enc, stride, dropout)
        self.decoder, self.decoder_drop = self.build_decoder(c_out, c_in, act_dec, stride, dropout)

    @staticmethod
    def build_encoder(c_in, c_out, activation, stride, dropout):
        layers = list()
        layers.append(nn.Conv1d(c_in, c_out, 3, stride, padding=1))

        if activation is not None:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation)

        return nn.Sequential(*layers), nn.Dropout(dropout)

    @staticmethod
    def build_decoder(c_in, c_out, activation, stride, dropout):
        layers = list()
        layers.append(nn.ConvTranspose1d(c_in, c_out, 3, stride, padding=1, output_padding=stride - 1))

        if activation is not None:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation)

        return nn.Sequential(*layers), nn.Dropout(dropout)

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

        sig_length, num_channels = data_shape
        stride = 2
        self.sae1 = SAE(num_channels, 16, stride, nn.GELU(), None, 0.0)  # reconstructs
        self.sae2 = SAE(16, 32, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae3 = SAE(32, 64, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae4 = SAE(64, 128, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae5 = SAE(128, 128, 1, nn.GELU(), nn.GELU(), 0.0)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4, self.sae5]

        self.last_layer_signal_length = int(sig_length / stride ** 4)
        self.last_layer_filters = self.sae_modules[-1].num_chan_out
        enc_output_length = self.last_layer_signal_length * self.last_layer_filters

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_output_length, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_size, embedding_size)
        )

        self.unflatten = nn.Sequential(
            nn.Linear(embedding_size, enc_output_length),
            nn.BatchNorm1d(enc_output_length),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)
        x = self.sae5.encoder(x)
        return self.flatten(x)

    def decoder(self, x):
        x = self.unflatten(x)
        x = x.view(-1, self.last_layer_filters, self.last_layer_signal_length)
        x = self.sae5.decoder(x)
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        return self.sae1.decoder(x)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
