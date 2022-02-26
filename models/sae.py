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
        layers.append(nn.Conv1d(c_in, c_out, 7, stride, padding=3))

        if activation is not None:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(activation)

        return nn.Sequential(*layers), nn.Dropout(dropout)

    @staticmethod
    def build_decoder(c_in, c_out, activation, stride, dropout):
        layers = list()
        layers.append(nn.ConvTranspose1d(c_in, c_out, 7, stride, padding=3, output_padding=stride - 1))

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
