import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from neuralnets.util.io import read_volume
from neuralnets.util.tools import sample_unlabeled_input, sample_labeled_input


class StandardDataset(data.Dataset):
    """
    Standard dataset of N 2D images

    :param data_path: path to the dataset
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, scaling=None, type='tif3d'):
        self.data_path = data_path
        self.scaling = scaling

        # load the data
        self.data = read_volume(data_path, type=type)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        # data is rescaled to [0,1]
        self.data = self.data / 255

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.data.shape[0]

    def _get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std


class StronglyLabeledStandardDataset(StandardDataset):
    """
    Strongly labeled dataset of N 2D images and pixel-wise labels

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, label_path, scaling=None, type='tif3d'):
        super().__init__(data_path, scaling=scaling, type=type)

        self.label_path = label_path

        # load labels
        self.labels = read_volume(label_path, type=type)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = \
                F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = self.data[i]
        target = self.labels[i]

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if target.min() == 255:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledStandardDataset(StandardDataset):
    """
    Unlabeled dataset of N 2D images

    :param data_path: path to the dataset
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, scaling=None, type='tif3d'):
        super().__init__(data_path, scaling=scaling, type=type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = self.data[i]

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input


class VolumeDataset(data.Dataset):
    """
    Dataset for volumes

    :param data_path: path to the dataset
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000, type='tif3d'):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch

        # load the data
        self.data = read_volume(data_path, type=type)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        # data is rescaled to [0,1]
        self.data = self.data / 255

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def _get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std


class StronglyLabeledVolumeDataset(VolumeDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000, type='tif3d'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, type=type)

        self.label_path = label_path

        # load labels
        self.labels = read_volume(label_path, type=type)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = \
                F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if target.min() == 255:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledVolumeDataset(VolumeDataset):
    """
    Dataset for unlabeled volumes

    :param data_path: path to the dataset
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    """

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000, type='tif3d'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, type=type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = sample_unlabeled_input(self.data, self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input


class MultiVolumeDataset(data.Dataset):
    """
    Dataset for multiple volumes

    :param data_path: list of paths to the datasets
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional types: list of types of the volume files (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional sampling_mode: allow for uniform balance in sampling or not ("uniform" or "random")
    """

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000, types=['tif3d'], sampling_mode='uniform'):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch
        self.sampling_mode = sampling_mode

        # load the data
        self.data = []
        self.data_sizes = []
        for k, path in enumerate(data_path):
            data = read_volume(path, type=types[k])

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(data.shape, scaling[k]), dtype=int)
                data = \
                    F.interpolate(torch.Tensor(data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                  mode='area')[0, 0, ...].numpy()

            # data is rescaled to [0,1]
            data = data / 255

            self.data.append(data)
            self.data_sizes.append(data.size)
        self.data_sizes = np.array(self.data_sizes)
        self.data_sizes = self.data_sizes / np.sum(self.data_sizes)

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def _get_stats(self):
        mu = []
        std = []

        for data in self.data:
            mu.append(np.mean(data))
            std.append(np.std(data))

        return mu, std


class StronglyLabeledMultiVolumeDataset(MultiVolumeDataset):
    """
    Dataset for multiple pixel-wise labeled volumes

    :param data_path: list of paths to the datasets
    :param data_path: list of paths to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional types: list of types of the volume files (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional sampling_mode: allow for uniform balance in sampling or not ("uniform" or "random")
    """

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000, types=['tif3d'],
                 sampling_mode='uniform'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, types=types,
                         sampling_mode=sampling_mode)

        self.label_path = label_path

        # load the data
        self.labels = []
        for k, path in enumerate(label_path):
            labels = read_volume(path, type=types[k])

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(labels.shape, scaling[k]), dtype=int)
                labels = \
                    F.interpolate(torch.Tensor(labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                  mode='area')[0, 0, ...].numpy()

            self.labels.append(labels)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # select dataset
        if self.sampling_mode is 'uniform':
            k = np.random.randint(0, len(self.data))
        else:
            k = np.random.choice(len(self.data), p=self.data_sizes)

        # get random sample
        input, target = sample_labeled_input(self.data[k], self.labels[k], self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if target.min() == 255:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledMultiVolumeDataset(MultiVolumeDataset):
    """
    Dataset for multiple unlabeled volumes

    :param data_path: list of paths to the datasets
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional types: list of types of the volume files (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional sampling_mode: allow for uniform balance in sampling or not ("uniform" or "random")
    """

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000, types='tif3d',
                 sampling_mode='uniform'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, types=types,
                         sampling_mode=sampling_mode)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # select dataset
        if self.sampling_mode is 'uniform':
            k = np.random.randint(0, len(self.data))
        else:
            k = np.random.choice(len(self.data), p=self.data_sizes)

        # get random sample
        input = sample_unlabeled_input(self.data[k], self.input_shape)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input
