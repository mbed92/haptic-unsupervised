import torch
from torchsummary import summary


class BackpropConfig:
    model: torch.nn.Module
    lr: float
    eta_min: float
    weight_decay: float
    epochs: int


def backprop_init(config: BackpropConfig):
    optimizer = torch.optim.AdamW(config.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)
    return optimizer, scheduler


def hardware_upload(model: torch.nn.Module, input_size: iter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=input_size)
    return device

