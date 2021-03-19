
import warnings

from neuralnets.util.tools import sample_unlabeled_input, sample_labeled_input, normalize
from neuralnets.util.augmentation import split_segmentation_transforms
from neuralnets.data.base import *


MAX_SAMPLING_ATTEMPTS = 20


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


def _map_cois(y, coi):
    """
    Maps the classes of interest to consecutive label indices

    :param y: labels
    :param coi: classes of interest
    :return: reindexed labels
    """
    coi_ = list(coi)
    coi_.sort()
    for i, c in enumerate(coi_):
        y[y == c] = i

    return y


class LabeledStandardDataset(StandardDataset):
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


class LabeledVolumeDataset(VolumeDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param labels: path to the labels or a 3D volume that has already been loaded, possible formats:
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional sampling_type: type of sampling in case of multiple datasets
            - joint: the dataset will generate random samples in each dataset and return all of them
            - single: the dataset will generate a random sample from a randomly selected dataset and return that
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, len_epoch=None, type='tif3d', coi=(0, 1),
                 in_channels=1, orientations=(0,), batch_size=1, data_dtype='uint8', label_dtype='uint8',
                 norm_type='unit', transform=None, range_split=None, range_dir=None, resolution=None,
                 match_resolution_to=None, sampling_type='joint'):
        super().__init__(data, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=data_dtype,
                         norm_type=norm_type, range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, sampling_type=sampling_type)

        if isinstance(labels, str) or isinstance(labels, np.ndarray):
            self.labels = [load_data(labels, data_type=type, dtype=label_dtype)]
        elif isinstance(labels, list) or isinstance(labels, tuple):  # list of data
            self.labels = []
            for labels_i in labels:
                self.labels.append(load_data(labels_i, data_type=type, dtype=label_dtype))
        else:
            raise ValueError('LabeledVolumeDataset requires labels in str, np.ndarray or list format')
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)

        # select a subset of slices of the data
        for i in range(len(self.labels)):
            if isinstance(range_split, list) and isinstance(range_dir, tuple):
                self.labels[i] = slice_subset(self.labels[i], range_split[i], range_dir[i])
            else:
                self.labels[i] = slice_subset(self.labels[i], range_split, range_dir)

        # rescale the dataset if necessary
        for i in range(len(self.labels)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.labels[self.match_resolution_to].shape, self.labels[i].shape)
                else:
                    if isinstance(self.scaling, list) or isinstance(self.scaling, tuple):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                labels_it = torch.Tensor(self.labels[i]).unsqueeze(0).unsqueeze(0)
                self.labels[i] = F.interpolate(labels_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

        # relabel classes of interest
        self.labels = [_map_cois(l, self.coi) for l in self.labels]

    def _sample(self, data_index):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[data_index].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x, y = sample_labeled_input(self.data[data_index], self.labels[data_index], input_shape)
        x = normalize(x, type=self.norm_type)
        y = y.astype(float)

        # reorient sample
        x = _orient(x, orientation=self.orientation)
        y = _orient(y, orientation=self.orientation)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x, y = x[np.newaxis, ...], y[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            y = y[c:c + 1]

        # augment sample
        if self.transform is not None:
            data = self.shared_transform(np.concatenate((x, y), axis=0))
            p = x.shape[0]
            x = self.x_transform(data[:p])
            y = self.y_transform(data[p:])

        # transform to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        return x, y

    def __getitem__(self, i, attempt=0):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        if self.sampling_type == 'single':

            # randomly select a dataset
            r = np.random.randint(len(self.data))

            # select a sample from dataset r
            x, y = self._sample(r)

            # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            if len(np.intersect1d(torch.unique(y).numpy(), self.coi)) == 0 and not self.warned:
                if attempt < MAX_SAMPLING_ATTEMPTS:
                    x, y = self.__getitem__(i, attempt=attempt + 1)
                else:
                    warnings.warn("No labeled pixels found after %d sampling attempts! " % attempt)
                    self.warned = True

        else:  # joint sampling

            xs = []
            ys = []

            for r in range(len(self.data)):

                # select a sample from dataset r
                x, y = self._sample(r)

                xs.append(x)
                ys.append(y)

            if len(self.data) == 1:
                x = xs[0]
                y = ys[0]
            else:
                x = xs
                y = ys

            # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            if np.sum([len(np.intersect1d(torch.unique(y_).numpy(), self.coi)) for y_ in ys]) == 0 and not self.warned:
                if attempt < MAX_SAMPLING_ATTEMPTS:
                    x, y = self.__getitem__(i, attempt=attempt + 1)
                else:
                    warnings.warn("No labeled pixels found after %d sampling attempts! " % attempt)
                    self.warned = True

        # return sample
        return x, y


class UnlabeledVolumeDataset(VolumeDataset):
    """
    Dataset for unlabeled volumes

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
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional sampling_type: type of sampling in case of multiple datasets
            - joint: the dataset will generate random samples in each dataset and return all of them
            - single: the dataset will generate a random sample from a randomly selected dataset and return that
    """

    def __init__(self, data, input_shape=None, scaling=None, len_epoch=None, type='tif3d', in_channels=1,
                 orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit', transform=None, range_split=None,
                 range_dir=None, resolution=None, match_resolution_to=None, sampling_type='joint'):
        super().__init__(data, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=dtype,
                         norm_type=norm_type, range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, sampling_type=sampling_type)

        self.transform = transform

    def _sample(self, data_index):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[data_index].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x = sample_unlabeled_input(self.data[data_index], input_shape)
        x = normalize(x, type=self.norm_type)

        # reorient sample
        x = _orient(x, orientation=self.orientation)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x = x[np.newaxis, ...]

        # augment sample
        if self.transform is not None:
            x = self.transform(x)

        # transform to tensors
        x = torch.from_numpy(x).float()

        return x

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        if self.sampling_type == 'single':

            # randomly select a dataset
            r = np.random.randint(len(self.data))

            # select a sample from dataset r
            x = self._sample(r)

        else:  # joint sampling

            xs = []

            for r in range(len(self.data)):

                # select a sample from dataset r
                x = self._sample(r)

                xs.append(x)

            if len(self.data) == 1:
                x = xs[0]
            else:
                x = xs

        # return sample
        return x


class LabeledSlidingWindowDataset(SlidingWindowDataset):
    """
    Dataset for pixel-wise labeled volumes with a sliding window

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param labels: path to the labels or a 3D volume that has already been loaded, possible formats:
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional coi: list or sequence of the classes of interest
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, type='tif3d', in_channels=1, orientations=(0,),
                 coi=(0, 1), batch_size=1, data_dtype='uint8', label_dtype='uint8', norm_type='unit', transform=None,
                 range_split=None, range_dir=None, resolution=None, match_resolution_to=None):
        super().__init__(data, input_shape, scaling=scaling, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=data_dtype, norm_type=norm_type,
                         range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to)

        if isinstance(labels, str) or isinstance(labels, np.ndarray):
            self.labels = [load_data(labels, data_type=type, dtype=label_dtype)]
        elif isinstance(labels, list) or isinstance(labels, tuple):  # list of data
            self.labels = []
            for labels_i in labels:
                self.labels.append(load_data(labels_i, data_type=type, dtype=label_dtype))
        else:
            raise ValueError('LabeledSlidingWindowDataset requires labels in str, np.ndarray or list format')
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)

        # select a subset of slices of the data
        for i in range(len(self.labels)):
            if isinstance(range_split, list) and isinstance(range_dir, tuple):
                self.labels[i] = slice_subset(self.labels[i], range_split[i], range_dir[i])
            else:
                self.labels[i] = slice_subset(self.labels[i], range_split, range_dir)

        # rescale the dataset if necessary
        for i in range(len(self.labels)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.labels[self.match_resolution_to].shape, self.labels[i].shape)
                else:
                    if isinstance(self.scaling, list) or isinstance(self.scaling, tuple):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                labels_it = torch.Tensor(self.labels[i]).unsqueeze(0).unsqueeze(0)
                self.labels[i] = F.interpolate(labels_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.labels = [pad2multiple(l, input_shape, value=255) for l in self.labels]

        # pad data so that additional channels can be sampled
        self.labels = [pad_channels(l, in_channels=in_channels, orientations=self.orientations) for l in self.labels]

        # relabel classes of interest
        self.labels = [_map_cois(l, self.coi) for l in self.labels]

    def __getitem__(self, i):

        # find dataset index
        r = 0
        szs = self.n_samples_dim.prod(axis=1)
        while szs[:r + 1].sum() <= i:
            r += 1

        # get spatial location
        j = i - szs[:r].sum()
        iz = j // (self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2])
        iy = (j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2]) // self.n_samples_dim[r, 2]
        ix = j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2] - iy * self.n_samples_dim[r, 2]
        pz = self.input_shape[0] * iz
        py = self.input_shape[1] * iy
        px = self.input_shape[2] * ix

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[r].shape, in_channels=self.in_channels)

        # get sample
        x, y = sample_labeled_input(self.data[r], self.labels[r], input_shape, zyx=(pz, py, px))
        x = normalize(x, type=self.norm_type)
        y = y.astype(float)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x, y = x[np.newaxis, ...], y[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            y = y[c:c + 1]

        # augment sample
        if self.transform is not None:
            data = self.shared_transform(np.concatenate((x, y), axis=0))
            p = x.shape[0]
            x = self.x_transform(data[:p])
            y = self.y_transform(data[p:])

        # transform to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
        if len(np.intersect1d(torch.unique(y).numpy(), np.arange(len(self.coi)))) == 0 and not self.warned:
            warnings.warn("No labeled pixels found! ")
            self.warned = True

        # return sample
        return x, y
