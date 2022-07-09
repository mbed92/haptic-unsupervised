import torch
import torch.nn as nn

from models.sae import SAE


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
