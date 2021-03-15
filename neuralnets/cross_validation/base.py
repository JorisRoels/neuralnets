
import os
import numpy as np
import torch
import pytorch_lightning as pl

from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from neuralnets.data.datasets import LabeledVolumeDataset
from neuralnets.networks.unet import UNet2D
from neuralnets.util.losses import CrossEntropyLoss


def _correct_type(param, values):

    vs = values.split(';')
    values = []
    for v in vs:
        try:
            v_ = float(v)
        except ValueError:
            v_ = v
        values.append(v_)

    return param, values


def parse_search_grid(sg_str):

    sg = sg_str.split('#')
    search_grid = {}
    for s in sg:
        param, values = s.split(':')
        param, values = _correct_type(param, values)
        search_grid[param] = values

    return search_grid


class PLClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50, log_refresh_rate=None,
                 train_batch_size=1, test_batch_size=1, num_workers=1, device=0, orientations=(0,),
                 normalization='unit', transform=None):

        # trainer parameters
        self.epochs = epochs
        self.gpus = gpus
        self.accelerator = accelerator
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.log_refresh_rate = log_refresh_rate

        # inference parameters
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.device = device  # temporary while multi GPU inference is not supported
        self.orientations = orientations
        self.normalization = normalization
        self.transform = transform

        # initialize the trainer
        checkpoint_callback = ModelCheckpoint(save_top_k=0)
        callbacks = [checkpoint_callback]
        self.trainer = pl.Trainer(max_epochs=self.epochs, gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # initialize the model: should be set in an implementation!
        self.model = None


    def fit(self, X, y):

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=self.model.input_shape, batch_size=self.train_batch_size,
                                     transform=self.transform)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self


    def predict(self, X, y):

        segmentation = self.model.segment(X, self.model, self.model.input_shape, in_channels=self.model.in_channels,
                                          batch_size=self.test_batch_size, device=self.device,
                                          orientations=self.orientations, normalization=self.normalization)

        return segmentation


    def score(self, X, y, sample_weight=None):

        # validate each model state and save the metrics
        checkpoints = os.listdir(self.trainer.checkpoint_callback.dirpath)
        checkpoints.sort()
        metrics = np.zeros(len(checkpoints))
        for i, ckpt in enumerate(checkpoints):
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, ckpt)
            self.model.load_state_dict(torch.load(ckpt_path))
            metrics[i] = self.model.validate(X, y, self.model.input_shape, in_channels=self.model.in_channels,
                                             classes_of_interest=self.model.coi, batch_size=self.test_batch_size,
                                             device=self.device, orientations=self.orientations,
                                             normalization=self.normalization, report=False)

        # find the best model state
        j = np.argmax(metrics)
        metric = metrics[j]

        # remove the remaining checkpoints
        for i, ckpt in enumerate(checkpoints):
            if i != j:
                ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, ckpt)
                os.remove(ckpt_path)

        return metric


class UNet2DClassifier(PLClassifier):

    def __init__(self, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50, log_refresh_rate=None,
                 train_batch_size=1, test_batch_size=1, num_workers=1, device=0, orientations=(0,),
                 normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1),
                 feature_maps=64, levels=4, skip_connections=True, residual_connections=False, norm='instance',
                 activation='relu', dropout_enc=0.0, dropout_dec=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3):
        super().__init__(epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform)

        # parameters
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.residual_connections = residual_connections
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.norm = norm
        self.activation = activation
        self.coi = coi
        self.loss_fn = loss_fn
        self.lr = lr

        # initialize model
        self.model = UNet2D(input_shape=input_shape, in_channels=in_channels, feature_maps=feature_maps, levels=levels,
                            skip_connections=skip_connections, residual_connections=residual_connections,
                            dropout_enc=dropout_enc, dropout_dec=dropout_dec, norm=norm, activation=activation, coi=coi,
                            loss_fn=loss_fn, lr=lr)

