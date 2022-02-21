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
    y_hat, data_dict = model(x)
    loss = criterion(y_hat, y_true)
    return y_hat, loss, data_dict["lv"]


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
    best_acc_test = 0
    y_pred_test, y_true_test = [], []
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(args.epochs):
            mean_loss, correct = 0.0, 0
            model.train(True)
            embeddings, labels = [], []

            # train loop
            for step, data in enumerate(train_dataloader):
                batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                out, loss, latent_vector = train(batch_data, batch_labels, model, criterion, optimizer)
                mean_loss += loss.item()
                correct += batch_hits(out, batch_labels)
                embeddings.extend(latent_vector)
                labels.extend(batch_labels)

            writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
            writer.add_scalar('accuracy/train', accuracy(correct, len(train_ds)), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            # run test loop
            mean_loss, correct = 0.0, 0
            model.train(False)
            y_pred, y_true = [], []
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
                    out, loss, data_dict = query(batch_data, batch_labels, model, criterion)
                    mean_loss += loss.item()
                    correct += batch_hits(out, batch_labels)

                    # update statistics
                    _, predicted = torch.max(out.data, 1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    y_true.extend(batch_labels.data.cpu().numpy())

            # calculate epoch accuracy
            epoch_accuracy = accuracy(correct, len(test_ds))
            if epoch_accuracy > best_acc_test:
                torch.save(model, os.path.join(writer.log_dir, 'test_model'))
                best_acc_test = epoch_accuracy
                results['test'] = best_acc_test
                y_pred_test = y_pred
                y_true_test = y_true
                print(f'Epoch {epoch}, test accuracy: {best_acc_test}')

            # update tensorboard
            writer.add_scalar('loss/test', mean_loss / len(test_ds), epoch)
            writer.add_scalar('accuracy/test', epoch_accuracy, epoch)

            # test
            # model.train(False)
            # with torch.no_grad():
            #     for step, data in enumerate(test_dataloader):
            #         batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
            #         _, _, latent_vector = query(batch_data, batch_labels, model, criterion)
            #         embeddings.extend(latent_vector)
            #         labels.extend(batch_labels)

            # update tensorboard
            # if epoch % args.projector_interval == 0:
            #     embeddings = torch.stack(embeddings, 0).detach().cpu().numpy()
            #     labels = torch.stack(labels, 0).detach().cpu().numpy()
            #     writer.add_embedding(embeddings, labels, global_step=epoch)

        utils_haptr.log.save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), data_shape)
        writer.flush()

    # save all statistics
    utils_haptr.log.save_dict(results, os.path.join(log_dir, 'results.txt'))

    with torch.no_grad():
        x_train, y_train = utils.dataset.get_total_data_from_dataloader(train_dataloader)
        x_test, y_test = utils.dataset.get_total_data_from_dataloader(test_dataloader)
        x = torch.cat([x_train, x_test], 0).cpu()
        y = torch.cat([y_train, y_test], 0)
        model.cpu()
        _, data_dict = model(x)

    utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'visualization_test'), data_dict["lv"], y, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/put_0.yaml")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
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
    parser.add_argument('--projector-interval', type=int, default=100)

    args, _ = parser.parse_known_args()
    main(args)
