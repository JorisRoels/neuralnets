"""
    This is a script that illustrates grid search cross validation to optimize parameters for a 2D U-Net
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os

from neuralnets.data.datasets import LabeledVolumeDataset
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm, save, mkdir
from neuralnets.util.tools import set_seed, parse_params
from neuralnets.cross_validation.base import UNet2DClassifier, parse_search_grid

from multiprocessing import freeze_support
from sklearn.model_selection import GridSearchCV

from torch.utils.tensorboard import SummaryWriter
from itertools import product


def log_hparams(gs, log_dir='logs'):
    mkdir(hparams_dir)
    parameters = gs.param_grid
    param_keys = list(parameters.keys())
    param_values = [v for v in parameters.values()]
    metrics = gs.cv_results_['mean_test_score']
    for j, params_val in enumerate(product(*param_values)):
        tb = SummaryWriter(log_dir=log_dir)
        p_dict = {key: params_val[i] for i, key in enumerate(param_keys)}
        m_dict = {'mIoU': metrics[j]}
        tb.add_hparams(p_dict, m_dict)
        tb.close()


if __name__ == '__main__':
    freeze_support()

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
    split = params['train_val_test_split']
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                         RandomDeformation(), AddNoise(sigma_max=0.05), CleanDeformedLabels(params['coi'])])
    train = LabeledVolumeDataset(params['data'], params['labels'], input_shape=input_shape, type=params['type'],
                                 batch_size=params['train_batch_size'], transform=transform, range_split=(0, split[0]),
                                 range_dir=params['split_orientation'])
    X_train = train.data
    y_train = train.labels

    """
        Build the network
    """
    print_frm('Building the network')
    clf = UNet2DClassifier(epochs=params['epochs'], gpus=params['gpus'], accelerator=params['accelerator'],
                           log_dir=params['log_dir'], log_freq=params['log_freq'],
                           log_refresh_rate=params['log_refresh_rate'], train_batch_size=params['train_batch_size'],
                           test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                           device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                           levels=params['levels'], dropout_enc=params['dropout'], dropout_dec=params['dropout'],
                           norm=params['norm'], activation=params['activation'], coi=params['coi'],
                           loss_fn=params['loss'])

    """
        Perform cross validation grid search
    """
    print_frm('Starting grid search cross validation')
    search_grid = parse_search_grid(params['search_grid'])
    gs = GridSearchCV(clf, search_grid, cv=params['folds'], verbose=4)
    gs.fit(X_train, y_train)

    """
        Save and report results
    """
    save(gs.cv_results_, params['results_file'])
    hparams_dir = os.path.join(params['log_dir'], 'hparams')
    log_hparams(gs, log_dir=hparams_dir)
    print_frm(gs.best_params_)
    print_frm('Best mIoU: %.6f' % gs.best_score_)

    print_frm('Finished!')
