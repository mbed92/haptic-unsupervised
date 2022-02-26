import torch.nn as nn

from models.sae import SAE


class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, data_shape: list):
        super().__init__()
        assert len(data_shape) > 0

        sig_length, num_channels = data_shape
        stride = 2
        self.sae1 = SAE(num_channels, 16, stride, nn.GELU(), None, 0.0)  # reconstructs
        self.sae2 = SAE(16, 32, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae3 = SAE(32, 64, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae4 = SAE(64, 3, stride, nn.GELU(), nn.GELU(), 0.0)
        self.sae_modules = [self.sae1, self.sae2, self.sae3, self.sae4]
        self.embedding_size = int(sig_length / (stride ** len(self.sae_modules)))

        # self attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=1, dropout=0.2)

    def encoder(self, x):
        x = self.sae1.encoder(x)
        x = self.sae2.encoder(x)
        x = self.sae3.encoder(x)
        x = self.sae4.encoder(x)
        x_attn, _ = self.attention_layer(x, x, x)
        return (x + x_attn).view(x.shape[0], -1)

    def decoder(self, x):
        x = x.view(x.shape[0], 3, -1)
        x = self.sae4.decoder(x)
        x = self.sae3.decoder(x)
        x = self.sae2.decoder(x)
        x = self.sae1.decoder(x)
        return x

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
