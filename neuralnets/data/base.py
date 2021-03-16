import numpy as np
import numpy.random as rnd
import torch
import torch.nn.functional as F
import torch.utils.data as data

from neuralnets.util.io import read_volume, print_frm


def _len_epoch(input_shape, data_shape, target=0.01, delta = 0.005, df=2, max_len=1000000, reps=20, D_max=100000):

    def _compute_target(d, D, reps, n):
        cnt = 0
        for i in range(reps):
            x = np.zeros((D))
            for k in range(n):
                j = rnd.randint(D - d)
                x[j:j+d] = 1
            cnt += np.mean(1-x)
        target = cnt / reps
        return target

    rnd.seed(0)

    d = np.prod(input_shape)
    D = np.prod(data_shape)

    r = np.ceil(np.power(D / D_max, 1 / 3))
    d = max(1, int(d / r / r / r))
    D = int(D / r / r / r)

    n = 1
    f = 1 + df
    t = 1
    while n < max_len:
        t = _compute_target(d, D, reps, n)
        if np.abs(t - target) < delta:
            return t, n
        elif t >= target + delta:
            n = min(max(n + 1, int(n * f)), max_len)
        else:
            n = max(min(n - 1, int(n / f)), 1)
            df = df / 2
            f = 1 + df

    return t, n


def pad2multiple(x, input_shape, value=0):

    pad_width = []
    for d in range(3):
        ts = (int(np.ceil(x.shape[d] / input_shape[d])) * input_shape[d]) - x.shape[d]
        pad_width.append((0, ts))
    pad_width = tuple(pad_width)

    x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=value)

    return x


def pad_channels(x, in_channels=1, orientations=(0,)):

    pad_width = []
    for d in range(3):
        ts = 0
        if d in orientations:
            ts = in_channels - 1
        pad_width.append((0, ts))
    pad_width = tuple(pad_width)

    x = np.pad(x, pad_width=pad_width, mode='edge')

    return x


def slice_subset(x, range, orientation):
    if range is not None and orientation is not None:
        start, stop = range
        z_dim, y_dim, x_dim = x.shape
        if orientation == 'z':
            start = int(start * z_dim)
            stop = int(stop * z_dim)
            x = x[start:stop, :, :]
        elif orientation == 'y':
            start = int(start * y_dim)
            stop = int(stop * y_dim)
            x = x[:, start:stop, :]
        else:
            start = int(start * x_dim)
            stop = int(stop * x_dim)
            x = x[:, :, start:stop]
    return x


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


class VolumeDataset(data.Dataset):
    """
    Dataset for volumes

    :param data: path to the dataset or a 3D volume that has already been loaded
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional range_dir: orientation of the slicing
    """

    def __init__(self, data, input_shape, scaling=None, len_epoch=None, type='tif3d', in_channels=1,
                 orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit', range_split=None, range_dir=None):
        if isinstance(data, str):
            self.data_path = data
            # load the data
            self.data = read_volume(data, type=type, dtype=dtype)
        else:
            self.data = data
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = len_epoch
        self.in_channels = in_channels
        self.orientations = orientations
        self.orientation = 0
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.range_split = range_split
        self.range_dir = range_dir

        # select a subset of slices of the data
        self.data = slice_subset(self.data, range_split, range_dir)

        # compute length epoch if necessary
        if len_epoch is None or len_epoch < 0:
            print_frm('Epoch length not set... estimating full coverage value')
            t, self.len_epoch = _len_epoch(self.input_shape, self.data.shape)
            print_frm('Epoch length automatically set to %d, this covers %.2f%% of the data on average'
                      % (self.len_epoch, (1-t)*100))

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


class SlidingWindowDataset(data.Dataset):
    """
    Dataset for sliding window over volumes

    :param data: path to the dataset or a 3D volume that has already been loaded
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional range_dir: orientation of the slicing
    """

    def __init__(self, data, input_shape, scaling=None, type='tif3d', in_channels=1, orientations=(0,), batch_size=1,
                 dtype='uint8', norm_type='unit', range_split=None, range_dir=None):
        if isinstance(data, str):
            self.data_path = data
            # load the data
            self.data = read_volume(data, type=type, dtype=dtype)
        else:
            self.data = data
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.orientations = orientations
        self.scaling = scaling
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.range_split = range_split
        self.range_dir = range_dir

        # select a subset of slices of the data
        self.data = slice_subset(self.data, range_split, range_dir)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.data.shape, scaling), dtype=int)
            self.data = \
                F.interpolate(torch.Tensor(self.data[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.data = pad2multiple(self.data, input_shape, value=0)

        # compute size of the data
        sz = 1
        self.n_samples_dim = []
        for d in range(3):
            self.n_samples_dim.append(self.data.shape[d] // self.input_shape[d])
            sz *= self.n_samples_dim[-1]
        self.n_samples = sz

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.data = pad_channels(self.data, in_channels=in_channels, orientations=self.orientations)

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.n_samples

    def _get_stats(self):
        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std


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
