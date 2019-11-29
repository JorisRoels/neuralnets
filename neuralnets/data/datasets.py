import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from neuralnets.util.io import read_tif
from neuralnets.util.tools import sample_unlabeled_input, sample_labeled_input


class VolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch

        # load the data
        self.data = read_tif(data_path)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        # normalize data
        self.data = self.data / 255

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std


class StronglyLabeledVolumeDataset(VolumeDataset):

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch)

        self.label_path = label_path

        # load labels
        self.labels = read_tif(label_path)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = \
                F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self.get_stats()

        # normalize labels
        self.labels = np.asarray(self.labels / 255, dtype='uint8')

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # make sure the targets are binary
        target = np.asarray(target > 0.5, dtype='uint8')

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target


class UnlabeledVolumeDataset(VolumeDataset):

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch)

        self.mu, self.std = self.get_stats()

    def __getitem__(self, i):

        # get random sample
        input = sample_unlabeled_input(self.data, self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input


class MultiVolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch

        # load the data
        self.data = []
        for k, path in enumerate(data_path):
            data = read_tif(path)

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(data.shape, scaling[k]), dtype=int)
                data = \
                F.interpolate(torch.Tensor(data[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[
                    0, 0, ...].numpy()

            # normalize data
            data = data/255

            self.data.append(data)

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def get_stats(self):
        mu = []
        std = []

        for data in self.data:
            mu.append(np.mean(data))
            std.append(np.std(data))

        return mu, std


class StronglyLabeledMultiVolumeDataset(MultiVolumeDataset):

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch)

        self.label_path = label_path

        # load the data
        self.labels = []
        for k, path in enumerate(label_path):
            labels = read_tif(path)

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(labels.shape, scaling[k]), dtype=int)
                labels = \
                F.interpolate(torch.Tensor(labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size), mode='area')[
                    0, 0, ...].numpy()

            # normalize labels
            labels = np.asarray(labels / 255, dtype='uint8')

            self.labels.append(labels)

        self.mu, self.std = self.get_stats()

    def __getitem__(self, i):

        # select dataset
        k = np.random.randint(0, len(self.data))

        # get random sample
        input, target = sample_labeled_input(self.data[k], self.labels[k], self.input_shape)

        # make sure the targets are binary
        target = np.asarray(target > 0.5, dtype='uint8')

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target


class UnlabeledMultiVolumeDataset(MultiVolumeDataset):

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch)

        self.mu, self.std = self.get_stats()

    def __getitem__(self, i):

        # select dataset
        k = np.random.randint(0, len(self.data))

        # get random sample
        input = sample_unlabeled_input(self.data[k], self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input
