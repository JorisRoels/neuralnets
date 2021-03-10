"""
    This is a script that illustrates training a Dense 2D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.networks.unet import DenseUNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed, parse_params
from neuralnets.util.losses import get_loss_function

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str, default='dense_unet_2d.yaml')
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
                         RandomDeformation(), AddNoise(sigma_max=0.05), CleanDeformedLabels(params['coi'])])
    train = LabeledVolumeDataset(params['train_data'], params['train_labels'], input_shape=input_shape,
                                 type=params['data_type'], batch_size=params['train_batch_size'], transform=transform)
    val = LabeledSlidingWindowDataset(params['val_data'], params['val_labels'], input_shape=input_shape,
                                      type=params['data_type'], batch_size=params['train_batch_size'])
    test = LabeledSlidingWindowDataset(params['test_data'], params['test_labels'], input_shape=input_shape,
                                       type=params['data_type'], batch_size=params['test_batch_size'])
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
    loss_fn = get_loss_function(params['loss'])
    net = DenseUNet2D(feature_maps=params['fm'], levels=params['levels'], dropout_enc=params['dropout'],
                      dropout_dec=params['dropout'], norm=params['norm'], activation=params['activation'],
                      coi=params['coi'], num_layers=params['num_layers'], k=params['k'], bn_size=params['bn_size'],
                      loss_fn=loss_fn)

    """
        Train the network
    """
    print_frm('Starting training')
    print_frm('Training with loss: %s' % loss_fn)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=params['epochs'], gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=[lr_monitor])
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network
    """
    print_frm('Testing network')
    trainer.test(net, test_loader)