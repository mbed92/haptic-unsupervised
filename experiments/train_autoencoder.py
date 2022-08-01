import argparse
import copy
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import submodules.haptic_transformer.utils as utils_haptr
import utils
from data import helpers
from models.autoencoders.conv import TimeSeriesConvAutoencoderConfig, TimeSeriesConvAutoencoder
from utils.ops import train_epoch, test_epoch

torch.manual_seed(42)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = helpers.load_dataset(config)
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # set up a model (find the best config)
    nn_params = TimeSeriesConvAutoencoderConfig()
    nn_params.data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    nn_params.dropout = args.dropout
    autoencoder = TimeSeriesConvAutoencoder(nn_params)
    device = utils.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # train the main autoencoder
    backprop_config = utils.ops.BackpropConfig()
    backprop_config.model = autoencoder
    backprop_config.optimizer = torch.optim.AdamW
    backprop_config.lr = args.lr
    backprop_config.eta_min = args.eta_min
    backprop_config.epochs = args.epochs
    backprop_config.weight_decay = args.weight_decay

    # train
    best_loss = 9999.9
    best_model = None
    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = utils.ops.backprop_init(backprop_config)

        for epoch in range(args.epochs):
            train_loss = train_epoch(autoencoder, main_train_dataloader, optimizer, device)
            writer.add_scalar(f'AE/train/{train_loss.name}', train_loss.get(), epoch)
            writer.add_scalar(f'AE/train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.flush()
            scheduler.step()

            # test epoch
            test_loss, exemplary_sample = test_epoch(autoencoder, main_test_dataloader, device)
            writer.add_scalar(f'AE/test/{test_loss.name}', test_loss.get(), epoch)
            writer.add_image('AE/test/image', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()

            # save the best autoencoder
            current_test_loss = test_loss.get()
            if current_test_loss < best_loss:
                torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))
                best_loss = current_test_loss
                best_model = copy.deepcopy(autoencoder)

    # verify the unsupervised classification accuracy
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()
            x_train, y_train = helpers.get_total_data_from_dataloader(main_train_dataloader)
            x_test, y_test = helpers.get_total_data_from_dataloader(main_test_dataloader)
            emb_train = best_model.encoder(x_train.permute(0, 2, 1)).numpy()
            emb_test = best_model.encoder(x_test.permute(0, 2, 1)).numpy()
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_train'), emb_train, y_train, writer)
            utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/put.yaml")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1300238908180922)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.0006863541995399743)
    parser.add_argument('--weight-decay', type=float, default=0.00018546443538449212)
    parser.add_argument('--eta-min', type=float, default=1e-4)
    args, _ = parser.parse_known_args()

    main(args)
