''' Based on Unsupervised Deep Embedding for Clustering Analysis '''
import torch.nn as nn


# class SAE(nn.Module):
#     def __init__(self, c_in: int, c_out: int, stride: int, act_enc: nn.Module, act_dec: nn.Module, dropout: float):
#         super().__init__()
#
#         self.num_chan_in = c_in
#         self.num_chan_out = c_out
#         self.stride = stride
#         self.dropout = dropout
#
#         self.encoder, self.encoder_drop = self.build_encoder(c_in, c_out, act_enc, stride, dropout)
#         self.decoder, self.decoder_drop = self.build_decoder(c_out, c_in, act_dec, stride, dropout)
#
#     @staticmethod
#     def build_encoder(c_in, c_out, activation, stride, dropout):
#         layers = list()
#         layers.append(nn.Conv1d(c_in, c_out, 7, stride, padding=3))
#
#         if activation is not None:
#             layers.append(nn.BatchNorm1d(c_out))
#             layers.append(activation)
#
#         return nn.Sequential(*layers), nn.Dropout(dropout)
#
#     @staticmethod
#     def build_decoder(c_in, c_out, activation, stride, dropout):
#         layers = list()
#         layers.append(nn.ConvTranspose1d(c_in, c_out, 7, stride, padding=3, output_padding=stride - 1))
#
#         if activation is not None:
#             layers.append(nn.BatchNorm1d(c_out))
#             layers.append(activation)
#
#         return nn.Sequential(*layers), nn.Dropout(dropout)
#
#     def forward(self, inputs):
#         x = self.encoder(self.encoder_drop(inputs))
#         return self.decoder(self.decoder_drop(x))
#
#     def set_dropout(self, rate):
#         assert 0.0 <= rate < 1.0
#         self.encoder_drop.p = rate
#         self.decoder_drop.p = rate


def conv1d(c_in, c_out, kernel=3, stride=2, padding=1, dropout=0.2, activation=nn.GELU()):
    layers = list()
    layers.append(nn.Conv1d(c_in, c_out, kernel, stride, padding))

    if activation is not None:
        layers.append(nn.BatchNorm1d(c_out))
        layers.append(nn.GELU())

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def deconv1d(c_in, c_out, kernel=7, stride=2, padding=3, dropout=0.2, activation=nn.GELU()):
    layers = list()
    layers.append(nn.ConvTranspose1d(c_in, c_out, kernel, stride, padding=padding, output_padding=stride - 1))

    if activation is not None:
        layers.append(nn.BatchNorm1d(c_out))
        layers.append(nn.GELU())

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, data_shape: list, embedding_size: int):
        super().__init__()
        assert len(data_shape) > 0

        # encoder
        self.conv1 = conv1d(data_shape[-1], 16)
        self.conv2 = conv1d(16, 32)
        self.conv3 = conv1d(32, 64)
        self.conv4 = conv1d(64, 128, activation=None)

        # latent representation
        self.last_layer_signal_length = int(data_shape[0] / 2 ** 4)
        self.last_layer_filters = 128
        enc_output_length = self.last_layer_signal_length * self.last_layer_filters

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_output_length, embedding_size)
        )

        self.unflatten = nn.Sequential(
            nn.Linear(embedding_size, enc_output_length),
            nn.BatchNorm1d(enc_output_length),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # decoder
        self.deconv4 = deconv1d(128, 64)
        self.deconv3 = deconv1d(64, 32)
        self.deconv2 = deconv1d(32, 16)
        self.deconv1 = deconv1d(16, data_shape[-1], activation=None)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return x

    def decoder(self, x):
        x = self.unflatten(x)
        x = x.view(-1, self.last_layer_filters, self.last_layer_signal_length)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))

        # sig_length, num_channels = data_shape
        # stride = 2
        # self.sae1 = SAE(num_channels, 16, stride, nn.GELU(), None, 0.0)  # reconstructs
        # self.sae2 = SAE(16, 32, stride, nn.GELU(), nn.GELU(), 0.0)
        # self.sae3 = SAE(32, 64, stride, nn.GELU(), nn.GELU(), 0.0)
        # self.sae4 = SAE(64, 128, stride, nn.GELU(), nn.GELU(), 0.0)
        # self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]
        # self.last_layer_signal_length = int(sig_length / stride ** len(self.sae_modules))
        # self.last_layer_filters = self.sae_modules[-1].num_chan_out
        # enc_output_length = self.last_layer_signal_length * self.last_layer_filters
        #
        # # self attention layer
        # self.attention_layer = nn.MultiheadAttention(embed_dim=self.last_layer_signal_length, num_heads=1, dropout=0.2)
        #
        # self.flatten = nn.Sequential(
        #     nn.Flatten(),
        #     nn.GELU(),
        #     nn.Linear(enc_output_length, embedding_size)
        # )
        #
        # self.unflatten = nn.Sequential(
        #     nn.Linear(2 * embedding_size, enc_output_length),
        #     nn.GELU()
        # )

    # def encoder(self, x):
    #     x = self.sae1.encoder(x)
    #     x = self.sae2.encoder(x)
    #     x = self.sae3.encoder(x)
    #     x = self.sae4.encoder(x)
    #     attn, _ = self.attention_layer(x, x, x)
    #     x_1 = self.flatten(attn)
    #     x_2 = self.flatten(x)
    #     return torch.cat([x_1, x_2], -1)

    # def decoder(self, x):
    #     x = self.unflatten(x)
    #     x = x.view(-1, self.last_layer_filters, self.last_layer_signal_length)
    #     x = self.sae4.decoder(x)
    #     x = self.sae3.decoder(x)
    #     x = self.sae2.decoder(x)
    #     return self.sae1.decoder(x)
