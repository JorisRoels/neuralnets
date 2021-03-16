
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

    :param data: path to the dataset or a 3D volume that has already been loaded
    :param labels: path to the labels or a 3D volume that has already been loaded
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
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional range_dir: orientation of the slicing
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, len_epoch=None, type='tif3d', coi=(0, 1),
                 in_channels=1, orientations=(0,), batch_size=1, data_dtype='uint8', label_dtype='uint8',
                 norm_type='unit', transform=None, range_split=None, range_dir=None):
        super().__init__(data, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=data_dtype,
                         norm_type=norm_type, range_split=range_split, range_dir=range_dir)

        if isinstance(labels, str):
            self.labels_path = labels
            # load labels
            self.labels = read_volume(labels, type=type, dtype=label_dtype)
        else:
            self.labels = labels
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)

        # select a subset of slices of the data
        self.labels = slice_subset(self.labels, range_split, range_dir)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                        mode='area')[0, 0, ...].numpy()

        # relabel classes of interest
        self.labels = _map_cois(self.labels, self.coi)

    def __getitem__(self, i, attempt=0):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x, y = sample_labeled_input(self.data, self.labels, input_shape)
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

        # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
        if len(np.intersect1d(torch.unique(y).numpy(), self.coi)) == 0:
            if attempt < MAX_SAMPLING_ATTEMPTS:
                x, y = self.__getitem__(i, attempt=attempt + 1)
            # else:
                # warnings.warn("No labeled pixels found after %d sampling attempts! ")


        # return sample
        return x, y


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


class LabeledSlidingWindowDataset(SlidingWindowDataset):
    """
    Dataset for pixel-wise labeled volumes with a sliding window

    :param data: path to the dataset
    :param labels: path to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional coi: list or sequence of the classes of interest
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, type='tif3d', in_channels=1, orientations=(0,),
                 coi=(0, 1), batch_size=1, data_dtype='uint8', label_dtype='uint8', norm_type='unit', transform=None,
                 range_split=None, range_dir=None):
        super().__init__(data, input_shape, scaling=scaling, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=data_dtype, norm_type=norm_type,
                         range_split=range_split, range_dir=range_dir)

        if isinstance(labels, str):
            self.labels_path = labels
            # load labels
            self.labels = read_volume(labels, type=type, dtype=label_dtype)
        else:
            self.labels = labels
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)

        # select a subset of slices of the data
        self.labels = slice_subset(self.labels, range_split, range_dir)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                                        mode='area')[0, 0, ...].numpy()

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.labels = pad2multiple(self.labels, input_shape, value=255)

        # pad data so that additional channels can be sampled
        self.labels = pad_channels(self.labels, in_channels=in_channels, orientations=self.orientations)

        # relabel classes of interest
        self.labels = _map_cois(self.labels, self.coi)

    def __getitem__(self, i):

        # get spatial location
        iz = i // (self.n_samples_dim[1] * self.n_samples_dim[2])
        iy = (i - iz * self.n_samples_dim[1] * self.n_samples_dim[2]) // self.n_samples_dim[2]
        ix = i - iz * self.n_samples_dim[1] * self.n_samples_dim[2] - iy * self.n_samples_dim[2]
        pz = self.input_shape[0] * iz
        py = self.input_shape[1] * iy
        px = self.input_shape[2] * ix

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels)

        # get sample
        x, y = sample_labeled_input(self.data, self.labels, input_shape, zyx=(pz, py, px))
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

        # # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
        # if len(np.intersect1d(torch.unique(y).numpy(), self.coi)) == 0:
        #     warnings.warn("No labeled pixels found! ")

        # return sample
        return x, y


class LabeledMultiVolumeDataset(MultiVolumeDataset):
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
