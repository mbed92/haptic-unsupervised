import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import models
import submodules.haptic_transformer.utils as utils_haptr
import utils
from utils.EmbeddingDataset import EmbeddingDataset

torch.manual_seed(42)


def query_embedding(data, device, shape, is_first_pass: bool):
    if is_first_pass:
        batch_data, _ = utils.dataset.load_samples_to_device(data, device)
        return batch_data.reshape([-1, shape])
    else:
        return data


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    flatten_data_shape = train_ds.signal_length * train_ds.mean.shape[-1]
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # setup a model
    model = models.TimeSeriesAutoencoder(data_shape, args.embed_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=data_shape)

    # start pretraining SAE autoencoders
    for i, sae in enumerate([model.sae1, model.sae2, model.sae3, model.sae4]):
        sae_log_dir = os.path.join(log_dir, f'sae{i}')
        with SummaryWriter(log_dir=sae_log_dir) as writer:
            optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
            criterion = nn.MSELoss()
            sae.set_dropout(0.2)

            for epoch in range(args.epochs):
                # Train SAE first
                sae.train(True)
                train_dataloader = main_train_dataloader
                test_dataloader = main_test_dataloader
                for step, data in enumerate(train_dataloader):
                    batch_data = query_embedding(data, device, flatten_data_shape, bool(i == 0))
                    optimizer.zero_grad()
                    y_hat = sae(batch_data)
                    loss = criterion(y_hat, batch_data)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    scheduler.step()
                    writer.add_scalar(f'loss/SAE{i}/train', loss.item(), epoch)

                # Test the SAE
                mean_loss = list()
                sae.set_dropout(0.0)  # reset dropout after pretraining
                sae.train(False)
                with torch.no_grad():
                    for step, data in enumerate(test_dataloader):
                        batch_data = query_embedding(data, device, flatten_data_shape, bool(i == 0))
                        y_hat = sae(batch_data)
                        mean_loss.append(criterion(y_hat, batch_data).item())
                writer.add_scalar('loss//SAE{i}/test', sum(mean_loss) / len(mean_loss), epoch)
                writer.flush()

            # prepare data for the previously trained SAE for the next SAE
            flatten_data_shape = batch_data.shape[-1]
            train_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, train_dataloader, device,
                                                                  [flatten_data_shape])
            test_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, test_dataloader, device,
                                                                 [flatten_data_shape])

    # train the main autoencoder
    main_log_dir = os.path.join(log_dir, f'full')
    with SummaryWriter(log_dir=main_log_dir) as writer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        criterion = nn.MSELoss()

        for epoch in range(args.epochs):
            model.train(True)
            for step, data in enumerate(main_test_dataloader):
                batch_data = query_embedding(data, device, flatten_data_shape, True)
                optimizer.zero_grad()
                y_hat = model(batch_data)
                loss = criterion(y_hat, batch_data)
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                writer.add_scalar('loss/AE/train', loss.item(), epoch)

            mean_loss = list()
            model.train(False)
            with torch.no_grad():
                for step, data in enumerate(test_dataloader):
                    batch_data = query_embedding(data, device, flatten_data_shape, True)
                    y_hat = model(batch_data)
                    mean_loss.append(criterion(y_hat, batch_data).item())
            writer.add_scalar('loss/AE/test', sum(mean_loss) / len(mean_loss), epoch)
            writer.flush()

        # save trained autoencoder
        torch.save(model, os.path.join(writer.log_dir, 'test_model'))

    # with SummaryWriter(log_dir=log_dir) as writer:
    # for epoch in range(args.epochs):
    #     mean_loss, correct = 0.0, 0
    #     model.train(True)
    #     embeddings, labels = [], []
    #
    #     # train loop
    #     for step, data in enumerate(train_dataloader):
    #         batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
    #         out, loss, latent_vector = train(batch_data, batch_labels, model, criterion, optimizer)
    #         mean_loss += loss.item()
    #         correct += batch_hits(out, batch_labels)
    #         embeddings.extend(latent_vector)
    #         labels.extend(batch_labels)

    #         writer.add_scalar('loss/train', mean_loss / len(train_ds), epoch)
    #         writer.add_scalar('accuracy/train', accuracy(correct, len(train_ds)), epoch)
    #         writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    #         scheduler.step()
    #
    #         # test
    #         model.train(False)
    #         with torch.no_grad():
    #             for step, data in enumerate(test_dataloader):
    #                 batch_data, batch_labels = utils.dataset.load_samples_to_device(data, device)
    #                 _, _, latent_vector = query(batch_data, batch_labels, model, criterion)
    #                 embeddings.extend(latent_vector)
    #                 labels.extend(batch_labels)
    #
    #         # update tensorboard
    #         if epoch % args.projector_interval == 0:
    #             embeddings = torch.stack(embeddings, 0).detach().cpu().numpy()
    #             labels = torch.stack(labels, 0).detach().cpu().numpy()
    #             writer.add_embedding(embeddings, labels, global_step=epoch)
    #
    #     utils_haptr.log.save_statistics(y_true_test, y_pred_test, model, os.path.join(log_dir, 'test'), data_shape)
    #     writer.flush()
    #
    # # save all statistics
    # utils_haptr.log.save_dict(results, os.path.join(log_dir, 'results.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/put.yaml")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=160)
    parser.add_argument('--nheads', type=int, default=4)
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--feed-forward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    parser.add_argument('--model-type', type=str, default='haptr')
    parser.add_argument('--projector-interval', type=int, default=100)

    args, _ = parser.parse_known_args()
    main(args)
