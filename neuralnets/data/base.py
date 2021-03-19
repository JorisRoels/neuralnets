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


def load_data(x, data_type='pngseq', dtype='uint8'):
    if isinstance(x, str):  # path to a single volume
        return read_volume(x, type=data_type, dtype=dtype)
    elif isinstance(x, np.ndarray):  # data already loaded
        return x
    else:
        raise ValueError('Datasets require data in str or np.ndarray format, received %s' % type(x))


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

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional sampling_type: type of sampling in case of multiple datasets
            - joint: the dataset will generate random samples in each dataset and return all of them
            - single: the dataset will generate a random sample from a randomly selected dataset and return that
    :param optional return_domain: return the domain id during iterating
    """

    def __init__(self, data, input_shape, scaling=None, len_epoch=None, type='tif3d', in_channels=1, orientations=(0,),
                 batch_size=1, dtype='uint8', norm_type='unit', range_split=None, range_dir=None, resolution=None,
                 match_resolution_to=None, sampling_type='joint', return_domain=False):

        if isinstance(data, str) or isinstance(data, np.ndarray):
            self.data = [load_data(data, data_type=type, dtype=dtype)]
        elif isinstance(data, list) or isinstance(data, tuple):  # list of data
            self.data = []
            for data_i in data:
                self.data.append(load_data(data_i, data_type=type, dtype=dtype))
        else:
            raise ValueError('LabeledVolumeDataset requires data in str, np.ndarray or list format')
        self.input_shape = input_shape
        self.scaling = scaling
        self.len_epoch = int(len_epoch) if len_epoch is not None else len_epoch
        self.in_channels = int(in_channels)
        if isinstance(orientations, str):
            self.orientations = [int(item) for item in orientations.split(',')]
        else:
            self.orientations = orientations
        self.orientation = 0
        self.batch_size = int(batch_size)
        self.norm_type = norm_type
        self.range_split = range_split
        self.range_dir = range_dir
        self.resolution = resolution
        self.match_resolution_to = match_resolution_to
        self.sampling_type = sampling_type
        self.return_domain = return_domain
        self.warned = False

        # select a subset of slices of the data
        for i in range(len(self.data)):
            if isinstance(range_split, list) or isinstance(range_split, tuple):
                self.data[i] = slice_subset(self.data[i], range_split[i], range_dir[i])
            else:
                self.data[i] = slice_subset(self.data[i], range_split, range_dir)

        # compute length epoch if necessary
        if len_epoch is None or len_epoch < 0:
            print_frm('Epoch length not set... estimating full coverage value')
            if self.sampling_type == 'single':
                tt, tle = 0, 0
                for d in self.data:
                    t, le = _len_epoch(self.input_shape, d.shape)
                    tt += t
                    tle += le
                t, self.len_epoch = tt / len(self.data), tle
            else:
                i_max = int(np.argmax([d.size for d in self.data]))
                t, self.len_epoch = _len_epoch(self.input_shape, self.data[i_max].shape)
            print_frm('Epoch length automatically set to %d, this covers %.2f%% of the largest data on average'
                      % (self.len_epoch, (1-t)*100))

        # rescale the dataset if necessary
        for i in range(len(self.data)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.data[self.match_resolution_to].shape, self.data[i].shape)
                else:
                    if isinstance(self.scaling, list) or isinstance(self.scaling, tuple):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                data_it = torch.Tensor(self.data[i]).unsqueeze(0).unsqueeze(0)
                self.data[i] = F.interpolate(data_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch

    def _select_orientation(self):
        self.orientation = np.random.choice(self.orientations)


class SlidingWindowDataset(data.Dataset):
    """
    Dataset for sliding window over volumes

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional return_domain: return the domain id during iterating
    """

    def __init__(self, data, input_shape, scaling=None, type='tif3d', in_channels=1, orientations=(0,), batch_size=1,
                 dtype='uint8', norm_type='unit', range_split=None, range_dir=None, resolution=None,
                 match_resolution_to=None, return_domain=False):

        if isinstance(data, str) or isinstance(data, np.ndarray):
            self.data = [load_data(data, data_type=type, dtype=dtype)]
        elif isinstance(data, list) or isinstance(data, tuple):  # list of data
            self.data = []
            for data_i in data:
                self.data.append(load_data(data_i, data_type=type, dtype=dtype))
        else:
            raise ValueError('SlidingWindowDataset requires data in str, np.ndarray or list format')
        self.input_shape = input_shape
        self.scaling = scaling
        self.in_channels = int(in_channels)
        if isinstance(orientations, str):
            self.orientations = [int(item) for item in orientations.split(',')]
        else:
            self.orientations = orientations
        self.orientation = 0
        self.batch_size = int(batch_size)
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.range_split = range_split
        self.range_dir = range_dir
        self.resolution = resolution
        self.match_resolution_to = match_resolution_to
        self.return_domain = return_domain
        self.warned = False

        # select a subset of slices of the data
        for i in range(len(self.data)):
            if isinstance(range_dir, list) or isinstance(range_dir, tuple):
                self.data[i] = slice_subset(self.data[i], range_split[i], range_dir[i])
            else:
                self.data[i] = slice_subset(self.data[i], range_split, range_dir)

        # rescale the dataset if necessary
        for i in range(len(self.data)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.data[self.match_resolution_to].shape, self.data[i].shape)
                else:
                    if isinstance(self.scaling, list):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                data_it = torch.Tensor(self.data[i]).unsqueeze(0).unsqueeze(0)
                self.data[i] = F.interpolate(data_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.data = [pad2multiple(d, input_shape, value=0) for d in self.data]

        # compute size of the data
        self.n_samples_dim = np.zeros((len(self.data), 3), dtype=int)
        for i, data_i in enumerate(self.data):
            for d in range(3):
                self.n_samples_dim[i, d] = data_i.shape[d] // self.input_shape[d]
        self.n_samples = self.n_samples_dim.prod(axis=1).sum()

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.data = [pad_channels(d, in_channels=in_channels, orientations=self.orientations) for d in self.data]

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.n_samples
