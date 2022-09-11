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
from data import EmbeddingDataset, helpers
from models.autoencoders.sae import TimeSeriesAutoencoderConfig, TimeSeriesAutoencoder
from models.autoencoders.ops import train_epoch, test_epoch

torch.manual_seed(42)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'stacked_autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = helpers.load_dataset(config)
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # set up a model (find the best config)
    nn_params = TimeSeriesAutoencoderConfig()
    nn_params.optimizer = torch.optim.AdamW
    nn_params.data_shape = train_ds.signal_length, train_ds.mean.shape[-1]
    nn_params.stride = 2
    nn_params.kernel = args.kernel_size
    nn_params.activation = nn.GELU()
    nn_params.dropout = args.dropout
    autoencoder = TimeSeriesAutoencoder(nn_params)
    device = models.autoencoders.ops.hardware_upload(autoencoder, nn_params.data_shape)

    # start pretraining SAE auto encoders
    if args.pretrain_sae:
        train_dataloader = copy.deepcopy(main_train_dataloader)
        test_dataloader = copy.deepcopy(main_test_dataloader)

        for i, sae in enumerate(autoencoder.sae_modules):

            # setup backprop config for the full AE
            backprop_config_sae = models.autoencoders.ops.BackpropConfig()
            backprop_config_sae.optimizer = torch.optim.AdamW
            backprop_config_sae.model = sae
            backprop_config_sae.lr = args.lr_sae
            backprop_config_sae.eta_min = args.eta_min_sae
            backprop_config_sae.epochs = args.epochs_sae
            backprop_config_sae.weight_decay = args.weight_decay_sae

            with SummaryWriter(log_dir=os.path.join(log_dir, f'sae{i}')) as writer:
                optimizer, scheduler = models.autoencoders.ops.backprop_init(backprop_config_sae)

                # run train/test epoch
                for epoch in range(args.epochs_sae):
                    train_loss = train_epoch(sae, train_dataloader, optimizer, device)
                    writer.add_scalar(f'SAE/train/{train_loss.name}', train_loss.get(), epoch)
                    writer.add_scalar(f'SAE/train/lr', optimizer.param_groups[0]['lr'], epoch)
                    writer.flush()
                    scheduler.step()

                    # test epoch
                    test_loss, _ = test_epoch(sae, test_dataloader, device)
                    writer.add_scalar(f'SAE/test/{test_loss.name}', test_loss.get(), epoch)
                    writer.flush()

                # prepare data for the previously trained SAE for the next SAE
                train_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, train_dataloader, device)
                test_dataloader = EmbeddingDataset.gather_embeddings(sae.encoder, test_dataloader, device)

    # train the main autoencoder
    backprop_config_ae = models.autoencoders.ops.BackpropConfig()
    backprop_config_ae.model = autoencoder
    backprop_config_ae.optimizer = torch.optim.AdamW
    backprop_config_ae.lr = args.lr_ae
    backprop_config_ae.eta_min = args.eta_min_ae
    backprop_config_ae.epochs = args.epochs_ae
    backprop_config_ae.weight_decay = args.weight_decay_ae

    # train
    best_loss = 9999.9
    best_model = None
    with SummaryWriter(log_dir=os.path.join(log_dir, 'full')) as writer:
        optimizer, scheduler = models.autoencoders.ops.backprop_init(backprop_config_ae)

        for epoch in range(args.epochs_ae):
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
            pred_train, pred_test = utils.clustering.kmeans(emb_train, emb_test, train_ds.num_classes)
            utils.clustering.print_clustering_accuracy(y_train, pred_train, y_test, pred_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/put.yaml")

    parser.add_argument('--epochs-sae', type=int, default=400)
    parser.add_argument('--epochs-ae', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1300238908180922)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--lr-sae', type=float, default=1e-3)
    parser.add_argument('--lr-ae', type=float, default=0.0006863541995399743)
    parser.add_argument('--weight-decay-sae', type=float, default=1e-3)
    parser.add_argument('--weight-decay-ae', type=float, default=0.00018546443538449212)
    parser.add_argument('--eta-min-sae', type=float, default=1e-4)
    parser.add_argument('--eta-min-ae', type=float, default=1e-4)
    parser.add_argument('--pretrain-sae', dest='pretrain_sae', action='store_true')
    parser.add_argument('--dont-pretrain-sae', dest='pretrain_sae', action='store_false')
    parser.set_defaults(pretrain_sae=True)
    args, _ = parser.parse_known_args()

    main(args)