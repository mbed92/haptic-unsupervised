import torch
import torch.nn as nn
from torchsummary import summary

from utils.metrics import Mean


class BackpropConfig:
    model: torch.nn.Module
    lr: float
    eta_min: float
    weight_decay: float
    epochs: int
    optimizer: type


def backprop_init(config: BackpropConfig):
    optimizer = config.optimizer(config.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)
    return optimizer, scheduler


def hardware_upload(model: torch.nn.Module, input_size: iter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=input_size)
    return device


def query(model, x):
    outputs = model(x.permute(0, 2, 1)).permute(0, 2, 1)
    loss = nn.MSELoss()(outputs, x)
    return outputs, loss


def train(model, inputs, optimizer):
    optimizer.zero_grad()
    outputs, loss = query(model, inputs)
    loss.backward()
    optimizer.step()
    return outputs, loss


def train_epoch(model, dataloader, optimizer, device):
    reconstruction_loss = Mean("Reconstruction Loss")
    model.train(True)

    for data in dataloader:
        batch_data, _ = data
        outputs, loss = train(model, batch_data.to(device).float(), optimizer)
        reconstruction_loss.add(loss.item())

    return reconstruction_loss


def test_epoch(model, dataloader, device, add_exemplary_sample=True):
    reconstruction_loss = Mean("Reconstruction Loss")
    exemplary_sample = None
    model.train(False)

    with torch.no_grad():
        for data in dataloader:
            batch_data, _ = data
            outputs, loss = query(model, batch_data.to(device).float())

            # gather the loss
            reconstruction_loss.add(loss.item())

            # add an exemplary sample
            if add_exemplary_sample and exemplary_sample is None:
                y_pred = outputs[0].detach().cpu().numpy()
                y_true = data[0][0].detach().cpu().numpy()
                exemplary_sample = [y_pred, y_true]

    return reconstruction_loss, exemplary_sample
