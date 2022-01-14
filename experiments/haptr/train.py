import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import submodules.haptic_transformer.utils as utils
from submodules.haptic_transformer.experiments.transformer.transformer_train import query, train

torch.manual_seed(42)


def main(args):
    log_dir = utils.log.logdir_name('./', args.model_type)
    utils.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = ...
    data_shape = train_ds.signal_length, train_ds.num_modalities, train_ds.mean.shape[-1]
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    results = {}

    # setup a model
    model = ...

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=data_shape)

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    w = torch.Tensor(train_ds.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # start
    best_epoch_accuracy = 0
    best_acc_test = 0
    y_pred_val, y_true_val = [], []
    y_pred_test, y_true_test = [], []

    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(args.epochs):
            mean_loss, correct = 0.0, 0
            model.train(True)

            # train loop
            for step, data in enumerate(train_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                out, loss = train(batch_data, batch_labels, model, criterion, optimizer)
                mean_loss += loss.item()

            # write to the tensorboard
            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            # test clustering
            mean_loss, correct = 0.0, 0
            model.train(False)

            # run test loop
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                    out, loss = query(batch_data, batch_labels, model, criterion)
                    mean_loss += loss.item()

            # update tensorboard
            writer.add_scalar('loss/test', mean_loss / len(test_ds), epoch)

        utils.log.save_statistics(y_true_val, y_pred_val, model, os.path.join(log_dir, 'val'), data_shape)
        utils.log.save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), data_shape)
        writer.flush()

    # save all statistics
    utils.log.save_dict(results, os.path.join(log_dir, 'results.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/touching.yaml")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=8)
    parser.add_argument('--feed-forward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--repetitions', type=int, default=300)
    parser.add_argument('--model-type', type=str, default='haptr')

    args, _ = parser.parse_known_args()
    main(args)
