"""
    This is a script that illustrates training a Dense 2D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os
import pytorch_lightning as pl

from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.networks.unet import DenseUNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed, parse_params
from neuralnets.util.validation import validate

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str, default='neuralnets/train/dense_unet_2d.yaml')
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
    print_frm('Label distribution: ')
    for i in range(len(params['coi'])):
        print_frm('    - Class %d: %.3f (train) - %.3f (val) - %.3f (test)' %
                  (train.label_stats[0][i][0], train.label_stats[0][i][1],
                   val.label_stats[0][i][1], test.label_stats[0][i][1]))
    print_frm('    - Unlabeled pixels: %.3f (train) - %.3f (val) - %.3f (test)' %
              (train.label_stats[0][-1][1], val.label_stats[0][-1][1], test.label_stats[0][-1][1]))

    """
        Build the network
    """
    print_frm('Building the network')
    net = DenseUNet2D(feature_maps=params['fm'], levels=params['levels'], dropout_enc=params['dropout'],
                      dropout_dec=params['dropout'], norm=params['norm'], activation=params['activation'],
                      coi=params['coi'], num_layers=params['num_layers'], k=params['k'], bn_size=params['bn_size'],
                      loss_fn=params['loss'])

    """
        Train the network
    """
    print_frm('Starting training')
    print_frm('Training with loss: %s' % params['loss'])
    checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, monitor='val/mIoU', mode='max')
    trainer = pl.Trainer(max_epochs=params['epochs'], gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=params['log_refresh_rate'])
    trainer.fit(net, train_loader, val_loader)

    """
        Validate the network
    """
    print_frm('Validating the network')
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    validate(net, test.data[0], test.get_original_labels()[0], params['input_size'], in_channels=params['in_channels'],
             classes_of_interest=params['coi'], batch_size=params['test_batch_size'],
             write_dir=os.path.join(params['log_dir'], 'test_segmentation'), track_progress=True,
             device=params['gpus'][0])
