"""
    This is a script that illustrates training a 2D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os

import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from neuralnets.data.datasets import LabeledVolumeDataset
from neuralnets.networks.unet import LITUNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.losses import get_loss_function
from neuralnets.util.tools import set_seed, load_net, parse_params
from neuralnets.util.validation import validate

"""
    Parse all the arguments
"""
print_frm('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", help="Path to the configuration file", type=str, default='unet_2d.yaml')
args = parser.parse_args()
with open(args.config) as file:
    params = parse_params(yaml.load(file, Loader=yaml.FullLoader))

"""
Fix seed (for reproducibility)
"""
set_seed(params['seed'])

"""
    Load the data
"""
print_frm('Loading data')
input_shape = (1, *(params['input_size']))
transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                     RandomDeformation(), AddNoise(sigma_max=0.05)])
train = LabeledVolumeDataset(params['train_data'], params['train_labels'], input_shape=input_shape,
                             len_epoch=params['len_epoch'], type=params['data_type'],
                             in_channels=params['in_channels'], batch_size=params['train_batch_size'],
                             orientations=params['orientations'], transform=transform)
val = LabeledVolumeDataset(params['val_data'], params['val_labels'], input_shape=input_shape,
                           len_epoch=params['len_epoch'], type=params['data_type'], in_channels=params['in_channels'],
                           batch_size=params['train_batch_size'], orientations=params['orientations'])
train_loader = DataLoader(train, batch_size=params['train_batch_size'], num_workers=params['num_workers'],
                          pin_memory=True)
val_loader = DataLoader(val, batch_size=params['test_batch_size'], num_workers=params['num_workers'], pin_memory=True)

"""
    Build the network
"""
print_frm('Building the network')
net = LITUNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                activation=params['activation'], coi=params['classes_of_interest'])

"""
    Train the network
"""
print_frm('Starting training')
trainer = pl.Trainer(max_epochs=params['epochs'], gpus=params['gpus'])
trainer.fit(net, train_loader, val_loader)

# """
#     Setup logging directory
# """
# print_frm('Setting up log directories')
# if not os.path.exists(args.log_dir):
#     os.mkdir(args.log_dir)
#
# """
#     Load the data
# """
# input_shape = (1, args.input_size[0], args.input_size[1])
# print_frm('Loading data')
# augmenter = Compose([ToFloatTensor(device=args.device), Rotate90(), FlipX(prob=0.5), FlipY(prob=0.5),
#                      ContrastAdjust(adj=0.1, include_segmentation=True),
#                      RandomDeformation_2D(input_shape[1:], grid_size=(64, 64), sigma=0.01, device=args.device,
#                                           include_segmentation=True),
#                      AddNoise(sigma_max=0.05, include_segmentation=True)])
# train = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EMBL/train'),
#                                      os.path.join(args.data_dir, 'EM/EMBL/train_labels'),
#                                      input_shape=input_shape, len_epoch=args.len_epoch, type='pngseq',
#                                      in_channels=args.in_channels, batch_size=args.train_batch_size,
#                                      orientations=args.orientations)
# test = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EMBL/test'),
#                                     os.path.join(args.data_dir, 'EM/EMBL/test_labels'),
#                                     input_shape=input_shape, len_epoch=args.len_epoch, type='pngseq',
#                                     in_channels=args.in_channels, batch_size=args.test_batch_size,
#                                     orientations=args.orientations)
# train_loader = DataLoader(train, batch_size=args.train_batch_size)
# test_loader = DataLoader(test, batch_size=args.test_batch_size)
#
# """
#     Build the network
# """
# print_frm('Building the network')
# net = UNet2D(in_channels=args.in_channels, feature_maps=args.fm, levels=args.levels, dropout_enc=args.dropout,
#              dropout_dec=args.dropout, norm=args.norm, activation=args.activation, coi=args.classes_of_interest)
#
# """
#     Setup optimization for training
# """
# print_frm('Setting up optimization for training')
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
#
# """
#     Train the network
# """
# print_frm('Starting training')
# net.train_net(train_loader, test_loader, loss_fn, optimizer, args.epochs, scheduler=scheduler,
#               augmenter=augmenter, print_stats=args.print_stats, log_dir=args.log_dir, device=args.device)
#
# """
#     Validate the trained network
# """
# validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
#          write_dir=os.path.join(args.log_dir, 'segmentation_final'), classes_of_interest=args.classes_of_interest,
#          val_file=os.path.join(args.log_dir, 'validation_final.npy'), in_channels=args.in_channels)
# net = load_net(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
# validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
#          write_dir=os.path.join(args.log_dir, 'segmentation_best'), classes_of_interest=args.classes_of_interest,
#          val_file=os.path.join(args.log_dir, 'validation_best.npy'), in_channels=args.in_channels)
#
# print_frm('Finished!')
