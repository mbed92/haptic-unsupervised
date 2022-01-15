import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import submodules.haptic_transformer.utils as utils_haptr
import utils
from submodules.haptic_transformer.experiments.transformer.transformer_train import accuracy, batch_hits
from submodules.haptic_transformer.models import HAPTR

torch.manual_seed(42)


def train(x, y_true, model, criterion, optimizer):
    optimizer.zero_grad()
    y_hat, loss, latent_vector = query(x, y_true, model, criterion)
    loss.backward()
    optimizer.step()
    return y_hat, loss, latent_vector


def query(x, y_true, model, criterion):
    y_hat, latent_vector = model(x)
    loss = criterion(y_hat, y_true)
    return y_hat, loss, latent_vector


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'haptr_runs')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, _, test_ds = utils.dataset.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    results = {}

    # setup a model
    model = HAPTR(train_ds.num_classes,
                  args.projection_dim, train_ds.signal_length, args.nheads, args.num_encoder_layers,
                  args.feed_forward, args.dropout, [data_shape[-1]], 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=data_shape)

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    w = torch.Tensor(train_ds.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # start
    y_pred_val, y_true_val = [], []
    y_pred_test, y_true_test = [], []

    with SummaryWriter(log_dir=log_dir) as writer:
        # train/validation
        for epoch in range(args.epochs):
            mean_loss, correct = 0.0, 0
            model.train(True)

            # train loop
            for step, data in enumerate(train_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                out, loss, latent_vector = train(batch_data, batch_labels, model, criterion, optimizer)
                mean_loss += loss.item()
                correct += batch_hits(out, batch_labels)

            # write to the tensorboard
            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('accuracy/train', accuracy(correct, len(train_ds)), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            # test clustering
            mean_loss, correct = 0.0, 0
            model.train(False)

            # run test loop
            y_pred, y_true = [], []
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                    out, loss, latent_vector = query(batch_data, batch_labels, model, criterion)
                    mean_loss += loss.item()
                    correct += batch_hits(out, batch_labels)

                    # update statistics
                    _, predicted = torch.max(out.data, 1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(batch_labels.data.cpu().numpy())

            # update tensorboard
            epoch_accuracy = accuracy(correct, len(test_ds))
            writer.add_scalar('loss/test', mean_loss / len(test_ds), epoch)
            writer.add_scalar('accuracy/test', epoch_accuracy, epoch)

        utils_haptr.log.save_statistics(y_true_val, y_pred_val, model, os.path.join(log_dir, 'val'), data_shape)
        utils_haptr.log.save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), data_shape)
        writer.flush()

    # save all statistics
    utils_haptr.log.save_dict(results, os.path.join(log_dir, 'results.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/touching_haptr.yaml")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=4)
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--feed-forward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--model-type', type=str, default='haptr')

    args, _ = parser.parse_known_args()
    main(args)
