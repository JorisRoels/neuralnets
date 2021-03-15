"""
    This is a script that illustrates training a 3D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.networks.unet import UNet3D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed, parse_params

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str, default='unet_3d.yaml')
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
    split = params['train_val_test_split']
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                         RandomDeformation(), AddNoise(sigma_max=0.05), CleanDeformedLabels(params['coi'])])
    train = LabeledVolumeDataset(params['data'], params['labels'], input_shape=input_shape, type=params['type'],
                                 batch_size=params['train_batch_size'], transform=transform, range_split=(0, split[0]),
                                 range_dir=params['split_orientation'])
    val = LabeledSlidingWindowDataset(params['data'], params['labels'], input_shape=input_shape, type=params['type'],
                                      batch_size=params['test_batch_size'], range_split=(split[0], split[1]),
                                      range_dir=params['split_orientation'])
    test = LabeledSlidingWindowDataset(params['data'], params['labels'], input_shape=input_shape, type=params['type'],
                                       batch_size=params['test_batch_size'], range_split=(split[1], 1),
                                       range_dir=params['split_orientation'])
    train_loader = DataLoader(train, batch_size=params['train_batch_size'], num_workers=params['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(test, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                             pin_memory=True)

    """
        Build the network
    """
    print_frm('Building the network')
    net = UNet3D(feature_maps=params['fm'], levels=params['levels'], dropout_enc=params['dropout'],
                 dropout_dec=params['dropout'], norm=params['norm'], activation=params['activation'], coi=params['coi'],
                 loss_fn=params['loss'])

    """
        Train the network
    """
    print_frm('Starting training')
    print_frm('Training with loss: %s' % params['loss'])
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=params['epochs'], gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=[lr_monitor],
                         progress_bar_refresh_rate=params['log_refresh_rate'])
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network
    """
    print_frm('Testing network')
    trainer.test(net, test_loader)
