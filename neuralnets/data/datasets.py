import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from neuralnets.util.io import read_volume
from neuralnets.util.tools import sample_unlabeled_input, sample_labeled_input, normalize


def _orient(data, orientation=0):
    """
    This function essentially places the desired orientation axis to that of the original Z-axis
    For example:
          (Z, Y, X) -> (Y, Z, X) for orientation=1
          (Z, Y, X) -> (X, Y, Z) for orientation=2
    Note that applying this function twice corresponds to the identity transform

    :param data: assumed to be of shape (Z, Y, X)
    :param orientation: 0, 1 or 2 (respectively for Z, Y or X axis)
    :return: reoriented data sample
    """
    if orientation == 1:
        return np.transpose(data, axes=(1, 0, 2))
    elif orientation == 2:
        return np.transpose(data, axes=(2, 1, 0))
    else:
        return data


def _validate_shape(input_shape, data_shape, orientation=0, in_channels=1, levels=4):
    """
    Validates an input for propagation through a U-Net by taking into account the following:
        - Sampling along different orientations
        - Sampling multiple adjacent slices as channels
        - Maximum size that can be sampled from the data

    :param input_shape: original shape of the sample (Z, Y, X)
    :param data_shape: shape of the data to sample from (Z, Y, X)
    :param orientation: orientation to sample along
    :param in_channels: sample multiple adjacent slices as channels
    :param levels: amount of pooling layers in the network
    :return: the validated input shape
    """

    # make sure input shape can be edited
    input_shape = list(input_shape)

    # sample adjacent slices if necessary
    is2d = input_shape[0] == 1
    if is2d and in_channels > 1:
        input_shape[0] = in_channels

    # transform the data shape and input shape according to the orientation
    if orientation == 1:  # (Z, Y, X) -> (Y, Z, X)
        input_shape = [input_shape[1], input_shape[0], input_shape[2]]
    elif orientation == 2:  # (Z, Y, X) -> (X, Y, Z)
        input_shape = [input_shape[2], input_shape[1], input_shape[0]]

    # make sure the input shape fits in the data shape: i.e. find largest k such that n of the form n=k*2**levels
    for d in range(3):
        if not (is2d and d == orientation) and input_shape[d] > data_shape[d]:
            # 2D case: X, Y - 3D case: X, Y, Z
            # note we assume that the data has a least in_channels elements in each dimension
            input_shape[d] = int((data_shape[d] // (2 ** levels)) * (2 ** levels))

    # and return as a tuple
    return tuple(input_shape)


class StandardDataset(data.Dataset):
    """
    Standard dataset of N 2D images

    :param data_path: path to the dataset
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, scaling=None, type='tif3d', dtype='uint8', norm_type='unit'):
        self.data_path = data_path
        self.scaling = scaling
        self.norm_type = norm_type

        # load the data
        self.data = read_volume(data_path, type=type, dtype=dtype)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

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
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, label_path, scaling=None, type='tif3d', data_dtype='uint8', label_dtype='uint8',
                 coi=(0, 1), norm_type='unit'):
        super().__init__(data_path, scaling=scaling, type=type, dtype=data_dtype, norm_type=norm_type)

        self.label_path = label_path
        self.coi = coi

        # load labels
        self.labels = read_volume(label_path, type=type, dtype=label_dtype)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = \
                F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = normalize(self.data[i], type=self.norm_type)
        target = self.labels[i]

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if len(np.intersect1d(np.unique(target),
                              self.coi)) == 0:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledStandardDataset(StandardDataset):
    """
    Unlabeled dataset of N 2D images

    :param data_path: path to the dataset
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, scaling=None, type='tif3d', dtype='uint8', norm_type='unit'):
        super().__init__(data_path, scaling=scaling, type=type, dtype=dtype, norm_type=norm_type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = normalize(self.data[i], type=self.norm_type)

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
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000, type='tif3d', in_channels=1,
                 orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit'):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch
        self.in_channels = in_channels
        self.orientations = orientations
        self.orientation = 0
        self.batch_size = batch_size
        self.norm_type = norm_type

        # load the data
        self.data = read_volume(data_path, type=type, dtype=dtype)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def _get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

    def _select_orientation(self):
        self.orientation = np.random.choice(self.orientations)


class StronglyLabeledVolumeDataset(VolumeDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000, type='tif3d', coi=(0, 1),
                 in_channels=1, orientations=(0,), batch_size=1, data_dtype='uint8', label_dtype='uint8',
                 norm_type='unit'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=data_dtype,
                         norm_type=norm_type)

        self.label_path = label_path
        self.coi = coi

        # load labels
        self.labels = read_volume(label_path, type=type, dtype=label_dtype)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                        mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)
        target = _orient(target, orientation=self.orientation)

        if self.input_shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if len(np.intersect1d(np.unique(target),
                              self.coi)) == 0:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
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
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000, type='tif3d', in_channels=1,
                 orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=dtype,
                         norm_type=norm_type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input = sample_unlabeled_input(self.data, input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)

        if self.input_shape[0] > 1:
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
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional sampling_mode: allow for uniform balance in sampling or not ("uniform" or "random")
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, input_shape, scaling=None, len_epoch=1000, types=['tif3d'], sampling_mode='uniform',
                 in_channels=1, orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit'):
        self.data_path = data_path
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch
        self.sampling_mode = sampling_mode
        self.in_channels = in_channels
        self.orientations = orientations
        self.orientation = 0
        self.k = 0
        self.batch_size = batch_size
        self.norm_type = norm_type

        # load the data
        self.data = []
        self.data_sizes = []
        for k, path in enumerate(data_path):
            data = read_volume(path, type=types[k], dtype=dtype)

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(data.shape, scaling[k]), dtype=int)
                data = \
                    F.interpolate(torch.Tensor(data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                  mode='area')[0, 0, ...].numpy()

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

    def _select_orientation(self):
        self.orientation = np.random.choice(self.orientations)

    def _select_dataset(self):
        if self.sampling_mode == 'uniform':
            self.k = np.random.randint(0, len(self.data))
        else:
            self.k = np.random.choice(len(self.data), p=self.data_sizes)


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
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, label_path, input_shape=None, scaling=None, len_epoch=1000, types=['tif3d'],
                 sampling_mode='uniform', coi=(0, 1), in_channels=1, orientations=(0,), batch_size=1,
                 data_dtype='uint8', label_dtype='uint8', norm_type='unit'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, types=types,
                         sampling_mode=sampling_mode, in_channels=in_channels, orientations=orientations,
                         batch_size=batch_size, dtype=data_dtype, norm_type=norm_type)

        self.label_path = label_path
        self.coi = coi

        # load the data
        self.labels = []
        for k, path in enumerate(label_path):
            labels = read_volume(path, type=types[k], dtype=label_dtype)

            # rescale the dataset if necessary
            if scaling is not None:
                target_size = np.asarray(np.multiply(labels.shape, scaling[k]), dtype=int)
                labels = F.interpolate(torch.Tensor(labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                       mode='area')[0, 0, ...].numpy()

            self.labels.append(labels)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_dataset()
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[self.k].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input, target = sample_labeled_input(self.data[self.k], self.labels[self.k], input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)
        target = _orient(target, orientation=self.orientation)

        if self.input_shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if len(np.intersect1d(np.unique(target),
                              self.coi)) == 0:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
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
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional sampling_mode: allow for uniform balance in sampling or not ("uniform" or "random")
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, input_shape=None, scaling=None, len_epoch=1000, types='tif3d',
                 sampling_mode='uniform', in_channels=1, orientations=(0,), batch_size=1, dtype='uint8',
                 norm_type='unit'):
        super().__init__(data_path, input_shape, scaling=scaling, len_epoch=len_epoch, types=types,
                         sampling_mode=sampling_mode, in_channels=in_channels, orientations=orientations,
                         batch_size=batch_size, dtype=dtype, norm_type=norm_type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_dataset()
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[self.k].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        input = sample_unlabeled_input(self.data[self.k], input_shape)
        input = normalize(input, type=self.norm_type)

        # reorient sample
        input = _orient(input, orientation=self.orientation)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input
