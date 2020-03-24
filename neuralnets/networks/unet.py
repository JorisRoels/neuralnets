import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from neuralnets.networks.blocks import UNetConvBlock2D, UNetUpSamplingBlock2D, UNetConvBlock3D, UNetUpSamplingBlock3D
from neuralnets.util.losses import boundary_weight_map
from neuralnets.util.metrics import jaccard, accuracy_metrics
from neuralnets.util.tools import module_to_device, tensor_to_device, log_scalars, log_images_2d, log_images_3d, \
    augment_samples


class UNetEncoder2D(nn.Module):
    """
    2D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super(UNetEncoder2D, self).__init__()

        self.features = nn.Sequential()
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock2D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetDecoder2D(nn.Module):
    """
    2D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, norm='instance', dropout=0.0,
                 activation='relu'):
        super(UNetDecoder2D, self).__init__()

        self.features = nn.Sequential()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections:
                conv_block = UNetConvBlock2D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock2D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNet2D(nn.Module):
    """
    2D U-Net

    :param optional in_channels: number of input channels
    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout_enc: encoder dropout factor
    :param optional dropout_dec: decoder dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional bnd_weight_map: use boundary weight maps in training
    """

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, skip_connections=True, norm='instance',
                 activation='relu', dropout_enc=0.0, dropout_dec=0.0, bnd_weight_map=False):
        super(UNet2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.bnd_weight_map = bnd_weight_map
        self.encoder_outputs = None
        self.decoder_outputs = None

        # contractive path
        self.encoder = UNetEncoder2D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)
        # expansive path
        self.decoder = UNetDecoder2D(out_channels, feature_maps=feature_maps, levels=levels,
                                     skip_connections=skip_connections, norm=norm, dropout=dropout_dec,
                                     activation=activation)

    def forward(self, inputs):

        # contractive path
        self.encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        self.decoder_outputs, outputs = self.decoder(final_output, self.encoder_outputs)

        return outputs

    def train_epoch(self, loader, loss_fn, optimizer, epoch, augmenter=None, print_stats=1, writer=None,
                    write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader: dataloader
        :param loss_fn: loss function
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter: data augmenter
        :param print_stats: frequency of printing statistics
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # transfer to suitable device
            data = tensor_to_device(data, device)

            # get the inputs and augment if necessary
            x, y = augment_samples(data, augmenter=augmenter)
            y = y.long()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.bnd_weight_map:
                weight = boundary_weight_map(y)
            else:
                weight = None
            loss = loss_fn(y_pred, y, weight=weight)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset) / loader.batch_size, loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f' % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_avg], ['train/' + s for s in ['loss-seg']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_pred = F.softmax(y_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x, y, y_pred], ['train/' + s for s in ['x', 'y', 'y_pred']], writer, epoch=epoch)

        return loss_avg

    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False, device=0):
        """
        Tests the network for one epoch
        :param loader: dataloader
        :param loss_fn: loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average testing loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        y_preds = []
        ys = []
        for i, data in enumerate(loader):

            # get the inputs and transfer to suitable device
            x, y = tensor_to_device(data, device)
            x = x.float()
            y = y.long()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.bnd_weight_map:
                weight = boundary_weight_map(y)
            else:
                weight = None
            loss = loss_fn(y_pred, y, weight=weight)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            y_preds.append(F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 1, ...])
            ys.append(y[:, 0, ...].cpu().numpy())

        # compute interesting metrics
        y_preds = np.asarray(y_preds)
        ys = np.asarray(ys)
        j = jaccard(ys, y_preds)
        a, ba, p, r, f = accuracy_metrics(ys, y_preds)

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_avg, j, a, ba, p, r, f], ['test/' + s for s in
                                                        ['loss-seg', 'jaccard', 'accuracy', 'balanced-accuracy',
                                                         'precision', 'recall', 'f-score']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_pred = F.softmax(y_pred, dim=1)[:, 1:2, :, :].data
                log_images_2d([x, y, y_pred], ['test/' + s for s in ['x', 'y', 'y_pred']], writer, epoch=epoch)

        return loss_avg

    def train_net(self, train_loader, test_loader, loss_fn, optimizer, epochs, scheduler=None, test_freq=1,
                  augmenter=None, print_stats=1, log_dir=None, write_images_freq=1, device=0):
        """
        Trains the network
        :param train_loader: data loader with training data
        :param test_loader: data loader with testing data
        :param loss_fn: loss function
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter: data augmenter
        :param print_stats: frequency of logging statistics
        :param log_dir: logging directory
        :param write_images_freq: frequency of writing images
        :param device: GPU device where the computations should occur
        """
        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             augmenter=augmenter, print_stats=print_stats, writer=writer,
                             write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer,
                                            write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()


class UNetEncoder3D(nn.Module):
    """
    3D U-Net encoder

    :param optional in_channels: number of input channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super(UNetEncoder3D, self).__init__()

        self.features = nn.Sequential()
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock3D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock3D(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps, norm=norm,
                                           dropout=dropout, activation=activation)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs


class UNetDecoder3D(nn.Module):
    """
    3D U-Net decoder

    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, norm='instance', dropout=0.0,
                 activation='relu'):
        super(UNetDecoder3D, self).__init__()

        self.features = nn.Sequential()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections:
                conv_block = UNetConvBlock3D(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            else:
                conv_block = UNetConvBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                             norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv3d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i],
                                                                       outputs)  # also deals with concat
            else:
                outputs = getattr(self.features, 'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        outputs = self.output(outputs)

        return decoder_outputs, outputs


class UNet3D(nn.Module):
    """
    3D U-Net

    :param optional in_channels: number of input channels
    :param optional out_channels: number of output channels
    :param optional feature_maps: number of initial feature maps
    :param optional levels: levels of the encoder
    :param optional skip_connections: use skip connections or not
    :param optional norm: specify normalization ("batch", "instance" or None)
    :param optional dropout_enc: encoder dropout factor
    :param optional dropout_dec: decoder dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional bnd_weight_map: use boundary weight maps in training
    """

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, skip_connections=True, norm='instance',
                 activation='relu', dropout_enc=0.0, dropout_dec=0.0, bnd_weight_map=False):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.bnd_weight_map = bnd_weight_map
        self.encoder_outputs = None
        self.decoder_outputs = None

        # contractive path
        self.encoder = UNetEncoder3D(in_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                                     dropout=dropout_enc, activation=activation)
        # expansive path
        self.decoder = UNetDecoder3D(out_channels, feature_maps=feature_maps, levels=levels,
                                     skip_connections=skip_connections, norm=norm, dropout=dropout_dec,
                                     activation=activation)

    def forward(self, inputs):

        # contractive path
        self.encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        self.decoder_outputs, outputs = self.decoder(final_output, self.encoder_outputs)

        return outputs

    def train_epoch(self, loader, loss_fn, optimizer, epoch, augmenter=None, print_stats=1, writer=None,
                    write_images=False, device=0):
        """
        Trains the network for one epoch
        :param loader: dataloader
        :param loss_fn: loss function
        :param optimizer: optimizer for the loss function
        :param epoch: current epoch
        :param augmenter: data augmenter
        :param print_stats: frequency of printing statistics
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average training loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # transfer to suitable device
            data = tensor_to_device(data, device)

            # get the inputs and augment if necessary
            if augmenter is not None:
                x, y = augment_samples(data, augmenter=augmenter)
                y = y.long()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.bnd_weight_map:
                weight = boundary_weight_map(y)
            else:
                weight = None
            loss = loss_fn(y_pred, y, weight=weight)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset) / loader.batch_size, loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_avg], ['train/' + s for s in ['loss-seg']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_pred = F.softmax(y_pred, dim=1)[:, 1:2, :, :].data
                log_images_3d([x, y, y_pred], ['train/' + s for s in ['x', 'y', 'y_pred']], writer, epoch=epoch)

        return loss_avg

    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False, device=0):
        """
        Tests the network for one epoch
        :param loader: dataloader
        :param loss_fn: loss function
        :param epoch: current epoch
        :param writer: summary writer
        :param write_images: frequency of writing images
        :param device: GPU device where the computations should occur
        :return: average testing loss over the epoch
        """
        # perform training on GPU/CPU
        module_to_device(self, device)
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        y_preds = []
        ys = []
        for i, data in enumerate(loader):

            # get the inputs and transfer to suitable device
            x = tensor_to_device(data[0].float(), device)
            y = tensor_to_device(data[1].long(), device)

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.bnd_weight_map:
                weight = boundary_weight_map(y)
            else:
                weight = None
            loss = loss_fn(y_pred, y, weight=weight)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            y_preds.append(F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 1, ...])
            ys.append(y[:, 0, ...].cpu().numpy())

        # compute interesting metrics
        y_preds = np.asarray(y_preds)
        ys = np.asarray(ys)
        j = jaccard(ys, y_preds)
        a, ba, p, r, f = accuracy_metrics(ys, y_preds)

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f' % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            log_scalars([loss_avg, j, a, ba, p, r, f], ['test/' + s for s in
                                                        ['loss-seg', 'jaccard', 'accuracy', 'balanced-accuracy',
                                                         'precision', 'recall', 'f-score']], writer, epoch=epoch)

            # log images if necessary
            if write_images:
                y_pred = F.softmax(y_pred, dim=1)[:, 1:2, :, :].data
                log_images_3d([x, y, y_pred], ['test/' + s for s in ['x', 'y', 'y_pred']], writer, epoch=epoch)

        return loss_avg

    def train_net(self, train_loader, test_loader, loss_fn, optimizer, epochs, scheduler=None, test_freq=1,
                  augmenter=None, print_stats=1, log_dir=None, write_images_freq=1, device=0):
        """
        Trains the network
        :param train_loader: data loader with training data
        :param test_loader: data loader with testing data
        :param loss_fn: loss function
        :param optimizer: optimizer for the loss function
        :param epochs: number of training epochs
        :param scheduler: optional scheduler for learning rate tuning
        :param test_freq: frequency of testing
        :param augmenter: data augmenter
        :param print_stats: frequency of logging statistics
        :param log_dir: logging directory
        :param write_images_freq: frequency of writing images
        :param device: GPU device where the computations should occur
        """
        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             augmenter=augmenter, print_stats=print_stats, writer=writer,
                             write_images=epoch % write_images_freq == 0, device=device)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer,
                                            write_images=True, device=device)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()
