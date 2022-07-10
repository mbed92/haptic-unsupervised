import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import submodules.haptic_transformer.utils as utils_haptr
import utils
from models.autoencoder import LSTMAutoencoderConfig
from train_stacked_autoencoder import train_epoch, test_epoch

torch.manual_seed(42)


def main(args):
    log_dir = utils_haptr.log.logdir_name('./', 'lstm_autoencoder')
    utils_haptr.log.save_dict(args.__dict__, os.path.join(log_dir, 'args.txt'))

    # load data
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)
    main_train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    main_test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # set up a model (find the best config)
    nn_params = LSTMAutoencoderConfig()
    nn_params.input_num_channels = train_ds.mean.shape[-1]
    nn_params.sequence_length = train_ds.signal_length
    nn_params.latent_dim = 30
    nn_params.num_layers = 10
    nn_params.kernel = 3
    nn_params.dropout = 0.3
    autoencoder = models.LSTMAutoencoder(nn_params)
    device = utils.ops.hardware_upload(autoencoder, (nn_params.sequence_length, nn_params.input_num_channels))

    # train the main autoencoder
    backprop_config = utils.ops.BackpropConfig()
    backprop_config.model = autoencoder
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
            train_epoch_loss = train_epoch(autoencoder, main_train_dataloader, optimizer, device, clip_grad_norm=True)
            writer.add_scalar('loss/train/AE', train_epoch_loss, epoch)
            writer.add_scalar('lr/train/AE', optimizer.param_groups[0]['lr'], epoch)
            for name, module in autoencoder.named_children():
                norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in module.parameters()]), 2)
                writer.add_scalar(f'grad/train/AE/{name}', norm, epoch)

            writer.flush()
            scheduler.step()

            # test epoch
            test_epoch_loss, exemplary_sample = test_epoch(autoencoder, main_test_dataloader, device)
            writer.add_scalar('loss/test/AE', test_epoch_loss, epoch)
            writer.add_image('image/test/AE', utils.clustering.create_img(*exemplary_sample), epoch)
            writer.flush()

            # save the best autoencoder
            if test_epoch_loss < best_loss:
                torch.save(autoencoder, os.path.join(writer.log_dir, 'test_model'))
                best_loss = test_epoch_loss
                best_model = autoencoder

    # verify the unsupervised classification accuracy
    if best_model is not None:
        with torch.no_grad():
            best_model.cpu()
            x_train, y_train = utils.dataset.get_total_data_from_dataloader(main_train_dataloader)
            x_test, y_test = utils.dataset.get_total_data_from_dataloader(main_test_dataloader)
            emb_train = best_model.encoder(x_train.permute(0, 2, 1)).numpy()
            emb_test = best_model.encoder(x_test.permute(0, 2, 1)).numpy()
            for c in range(2, train_ds.num_classes):
                print(f"Clustering accuracy for {c} expected clusters.")
                pred_train, pred_test = utils.clustering.kmeans(emb_train, emb_test, c)
                utils.clustering.print_clustering_accuracy(y_train, pred_train, y_test, pred_test)
                print(f"\n")

        utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_train'), emb_train, y_train, writer)
        utils.clustering.save_embeddings(os.path.join(writer.log_dir, 'vis_test'), emb_test, y_test, writer, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic-unsupervised/config/unsupervised/touching.yaml")
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--lr', type=float, default=0.0009455447264165677)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eta-min', type=float, default=1e-4)

    args, _ = parser.parse_known_args()
    main(args)
