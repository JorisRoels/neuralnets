import copy
import random

import numpy as np
import torch
import torchvision.utils as vutils
from scipy.ndimage.morphology import binary_opening

from neuralnets.util.io import read_volume


def sample_labeled_input(data, labels, input_shape, preloaded=True, type='pngseq', data_shape=None):
    """
    Generate an input and target sample of certain shape from a labeled dataset

    :param data: data to sample from (a 3D numpy array if preloaded, a directory containing the data else)
    :param labels: labels to sample from (a 3D numpy array if preloaded, a directory containing the data else)
    :param input_shape: (z, x, y) shape of the sample
    :param preloaded: boolean that specifies whether the data is already in RAM
    :param type: type of the dataset that should be loaded in RAM (only necessary if preloaded==False)
    :param data_shape: (z, x, y) shape of the dataset to sample from (only necessary if preloaded==False)
    :return: a random sample
    """
    # randomize seed
    np.random.seed()

    # extract input and target patch
    if preloaded:  # if preloaded, we can simply load it from RAM
        # generate random position
        z = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
        x = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
        y = np.random.randint(0, data.shape[2] - input_shape[2] + 1)

        input = data[z:z + input_shape[0], x:x + input_shape[1], y:y + input_shape[2]]
        target = labels[z:z + input_shape[0], x:x + input_shape[1], y:y + input_shape[2]]
    else:  # if not preloaded, we have to additionally load it in RAM
        # generate random position
        z = np.random.randint(0, data_shape[0] - input_shape[0] + 1)
        x = np.random.randint(0, data_shape[1] - input_shape[1] + 1)
        y = np.random.randint(0, data_shape[2] - input_shape[2] + 1)

        input = read_volume(data, type=type, start=z, stop=z + input_shape[0])
        target = read_volume(labels, type=type, start=z, stop=z + input_shape[0])
        input = input[:, x:x + input_shape[1], y:y + input_shape[2]]
        target = target[:, x:x + input_shape[1], y:y + input_shape[2]]

    return copy.copy(input), copy.copy(target)


def sample_unlabeled_input(data, input_shape, preloaded=True, type='pngseq', data_shape=None):
    """
    Generate an input sample of certain shape from an unlabeled dataset

    :param data: data to sample from (a 3D numpy array if preloaded, a directory containing the data else)
    :param input_shape: (z, x, y) shape of the sample
    :param preloaded: boolean that specifies whether the data is already in RAM
    :param type: type of the dataset that should be loaded in RAM (only necessary if preloaded==False)
    :param data_shape: (z, x, y) shape of the dataset to sample from (only necessary if preloaded==False)
    :return: a random sample
    """
    # randomize seed
    np.random.seed()

    # extract input and target patch
    if preloaded:  # if preloaded, we can simply load it from RAM
        # generate random position
        z = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
        x = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
        y = np.random.randint(0, data.shape[2] - input_shape[2] + 1)

        input = data[z:z + input_shape[0], x:x + input_shape[1], y:y + input_shape[2]]
    else:  # if not preloaded, we have to additionally load it in RAM
        # generate random position
        z = np.random.randint(0, data_shape[0] - input_shape[0] + 1)
        x = np.random.randint(0, data_shape[1] - input_shape[1] + 1)
        y = np.random.randint(0, data_shape[2] - input_shape[2] + 1)

        input = read_volume(data, type=type, start=z, stop=z + input_shape[0])
        input = input[:, x:x + input_shape[1], y:y + input_shape[2]]

    return copy.copy(input)


def gaussian_window(size, sigma=1):
    """
    Returns a 3D Gaussian window that can be used for window weighting and merging

    :param size: size of the window
    :param sigma: standard deviation of the gaussian
    :return: the Gaussian window
    """
    # half window sizes
    hwz = size[0] // 2
    hwy = size[1] // 2
    hwx = size[2] // 2

    # construct mesh grid
    if size[0] % 2 == 0:
        axz = np.arange(-hwz, hwz)
    else:
        axz = np.arange(-hwz, hwz + 1)
    if size[1] % 2 == 0:
        axy = np.arange(-hwy, hwy)
    else:
        axy = np.arange(-hwy, hwy + 1)
    if size[2] % 2 == 0:
        axx = np.arange(-hwx, hwx)
    else:
        axx = np.arange(-hwx, hwx + 1)
    xx, zz, yy = np.meshgrid(axx, axz, axy)

    # normal distribution
    gw = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))

    # normalize so that the mask integrates to 1
    gw = gw / np.sum(gw)

    return gw


def load_unet(model_file, device=0):
    """
    Load a pretrained pytorch U-Net network

    :param model_file: path to the state dict checkpoint
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    :return: a module that corresponds to the trained network
    """
    from neuralnets.networks.unet import UNet2D, UNet3D

    if not torch.cuda.is_available() or device == 'cpu':
        state_dict = torch.load(model_file, map_location='cpu')
    else:
        state_dict = torch.load(model_file, map_location='cuda:' + str(device))

    # extract the hyperparameters of the network
    in_channels = state_dict['encoder.features.convblock1.conv1.unit.0.weight'].size(1)
    feature_maps = state_dict['encoder.features.convblock1.conv1.unit.0.weight'].size(0)
    out_channels = state_dict['decoder.output.weight'].size(0)
    coi = tuple(np.arange(0, out_channels))
    levels = int(list(state_dict.keys())[-3][len('encoder.features.convblock')])
    skip_connections = True
    norm = 'batch' if 'norm' in list(state_dict.keys())[2] else 'instance'
    activation = 'relu'
    dropout_enc = 0.0
    dropout_dec = 0.0

    # initialize the network
    if state_dict['encoder.features.convblock1.conv1.unit.0.weight'].dim() == 4:
        net = UNet2D(in_channels=in_channels, coi=coi, feature_maps=feature_maps, levels=levels,
                     skip_connections=skip_connections, norm=norm, activation=activation, dropout_enc=dropout_enc,
                     dropout_dec=dropout_dec)
    else:
        net = UNet3D(in_channels=in_channels, coi=coi, feature_maps=feature_maps, levels=levels,
                     skip_connections=skip_connections, norm=norm, activation=activation, dropout_enc=dropout_enc,
                     dropout_dec=dropout_dec)

    # load the parameters in the model
    net.load_state_dict(state_dict)

    # map to the correct device
    module_to_device(net, device=device)

    return net


def load_net(model_file, device=0):
    """
    Load a pretrained pytorch network, currently only support for U-Net

    :param model_file: path to the state dict checkpoint
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    :return: a module that corresponds to the trained network
    """
    return load_unet(model_file, device=device)


def save_net(model, model_file):
    """
    Save a pytorch network

    :param model: path to the state dict checkpoint
    :param model_file: path to the state dict checkpoint
    """
    torch.save(model.state_dict(), model_file)


def set_seed(seed):
    """
    Sets the seed of all randomized modules (useful for reproducibility)

    :param seed: seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_to_device(module, device):
    """
    Transfers a pytorch module to a specific GPU device

    :param module: module that should be transferred
    :param device: index of the device, or 'cpu' (if there are no GPU devices, it will be moved to the CPU)
    """
    if not torch.cuda.is_available() or device == 'cpu':
        module.cpu()
    else:
        module.cuda(device=device)


def tensor_to_device(x, device):
    """
    Transfers a pytorch tensor to a specific GPU device

    :param x: tensor or sequence/list of tensors that should be transferred
    :param device: index of the device (if there are no GPU devices or device is None, it will be moved to the CPU)
    :return x: same tensor, but switched to device
    """
    if isinstance(x, tuple) or isinstance(x, list):
        if not torch.cuda.is_available() or device is None:
            return [xx.cpu() for xx in x]
        else:
            return [xx.cuda(device=torch.device('cuda:' + str(device))) for xx in x]
    else:
        if not torch.cuda.is_available() or device is None:
            return x.cpu()
        else:
            return x.cuda(device=torch.device('cuda:' + str(device)))


def augment_samples(data, augmenter=None):
    """
    Augment a tensor with a specific augmenter

    :param data: tensor or sequence/list of tensors that should be augmented
    :param optional augmenter: augmenter that should be used (original data is returned if this is not specified)
    :return data: augmented tensor (or list of tensors)
    """
    if augmenter is not None:
        if isinstance(data, tuple) or isinstance(data, list):
            bs = [x.size(0) for x in data]
            data = [x.float() for x in data]
            data_aug = augmenter(torch.cat(data, dim=0)).float()
            return torch.split(data_aug, bs, dim=0)
        else:
            return augmenter(data.float()).float()
    return data


def get_labels(y, coi, dtype=int):
    """
    Maps general annotated image tensors to indexed labels for particular classes of interest

    :param y: annotated image tensor (B, N_1, N_2, ...)
    :param coi: classes of interest
    :param optional dtype: type of the tensor (typically integers)
    :return: indexed label tensor, ready for use in most loss functions (B, N_1, N_2, ...)
    """
    labels = torch.zeros_like(y, dtype=dtype)

    # convert labels to integers
    y = torch.round(y.float()).long()

    # loop over classes of interest
    for i, c in enumerate(coi):
        if i > 0:
            labels[y == c] = i

    # check if other classes are annotated, these can be labeled as background
    for c in torch.unique(y):
        if not c in coi and not c == 255:
            labels[y == c] = 0
    return labels


def get_unlabeled(y, dtype=int):
    """
    Maps general annotated image tensors to an indexed image of unlabeled pixels

    :param y: annotated image tensor (B, N_1, N_2, ...)
    :param coi: classes of interest
    :param optional dtype: type of the tensor (typically integers)
    :return: indexed label tensor, ready for use in most loss functions (B, N_1, N_2, ...)
    """
    unlabeled = torch.zeros_like(y, dtype=dtype)
    unlabeled[y == 255] = 1
    return unlabeled


def log_scalars(scalars, names, writer, epoch=0):
    """
    Writes a list of scalars to a tensorboard events file

    :param scalars: list of scalars (can be tensors or numpy arrays) that should be logged
    :param names: list of names that correspond to the scalars
    :param writer: writer used for logging
    :param epoch: current epoch
    """
    for name, scalar in zip(names, scalars):
        writer.add_scalar(name, scalar, epoch)


def log_images_2d(images, names, writer, epoch=0, scale_each=True):
    """
    Writes a list of 2D images to a tensorboard events file

    :param images: list of (2D) images (in pytorch tensor format, size [B, {1,3}, Y, X]) that should be logged
    :param names: list of names that correspond to the images
    :param writer: writer used for logging
    :param optional epoch: current epoch
    :param optional scale_each: scale each image or not
    """
    for id, x in zip(names, images):
        x = vutils.make_grid(x, normalize=x.max() - x.min() > 0, scale_each=scale_each)
        writer.add_image(id, x, epoch)


def log_images_3d(images, names, writer, epoch=0, scale_each=True):
    """
    Writes a list of 3D images to a tensorboard events file.
    For efficiency reasons, the center z-slice is selected from the 3D image

    :param images: list of (3D) images (in pytorch tensor format, size [B, {1,3}, Z, Y, X]) that should be logged
    :param names: list of names that correspond to the images
    :param writer: writer used for logging
    :param epoch: current epoch
    :param optional scale_each: scale each image or not
    """
    for id, x in zip(names, images):
        x = x[:, :, x.size(2) // 2, :, :]
        x = vutils.make_grid(x, normalize=x.max() - x.min() > 0, scale_each=scale_each)
        writer.add_image(id, x, epoch)


def clean_labels(y, n_classes):
    y_clean = y
    y = y.data.cpu().numpy()
    for b in range(y.shape[0]):
        y_b = y_clean[b, 0, ...]
        for c in range(n_classes):
            if not (c == 0 or c == y.shape[1] - 1):
                mask = binary_opening(y[b, 0, ...] == c)
                y_b[y_b == c] = 0
                y_b[torch.Tensor(mask).bool()] = c
        y_clean[b, 0, ...] = y_b
    return y_clean


def normalize(x, type=None, factor=None, mu=None, sigma=None):
    """
    Normalizes an numpy array

    :param x: an arbitrary numpy array
    :param type: the desired type of normalization (None, z, unit or minmax)
    :param factor: normalization factor (only if type is unit)
    :param mu: normalization mean (only if type is z)
    :param sigma: normalization std (only if type is z)
    :return: the normalized numpy array
    """
    if type == 'z':
        # apply z normalization
        if mu is None:
            mu = 0
        if sigma is None:
            sigma = 1
        return (x - mu) / sigma
    elif type == 'minmax':
        m = x.min()
        M = x.max()
        eps = 1e-5
        return (x - m + eps) / (M - m + eps)
    else:
        # apply unit normalization
        if factor == None:
            factors = {np.dtype('int8'): 2 ** 8 - 1,
                       np.dtype('uint8'): 2 ** 8 - 1,
                       np.dtype('int16'): 2 ** 16 - 1,
                       np.dtype('uint16'): 2 ** 16 - 1,
                       np.dtype('int32'): 2 ** 32 - 1,
                       np.dtype('uint32'): 2 ** 32 - 1,
                       np.dtype('int64'): 2 ** 64 - 1,
                       np.dtype('uint64'): 2 ** 64 - 1}
            if x.dtype in factors:
                factor = factors[x.dtype]
            else:
                factor = 1.
        return x / factor


def _find_split(y, test_size=0.33, axis=0):

    # reorient data
    if axis == 1:
        y_ = np.transpose(y, axes=(1, 0, 2))
    elif axis == 2:
        y_ = np.transpose(y, axes=(2, 1, 0))
    else:
        y_ = y

    start = 0
    stop = y_.shape[0] - 1
    n_total = np.sum(y_ != 255)

    while start != stop:

        # split halfway between start and stop
        split = (start + stop) // 2

        # compute test size
        n_test = np.sum(y_[split:] != 255)
        split_test_size = n_test / n_total

        if split == start:
            break

        if split_test_size < test_size:
            # test set needs to be larger
            stop = split
        else:
            # test set needs to be smaller
            start = split

    split = start

    return split


def _maximize_isotropy(shape, splits):

    # for each split and for both train and test volumes,
    # we compute the difference between a cubic hull and the actual volume
    maximum_covered_fraction = 0
    best_dim = 0
    for d, split in enumerate(splits):

        # volume train set
        train_shape = np.asarray(shape)
        train_shape[d] = split
        train_volume = np.prod(train_shape)
        train_cubic_hull_volume = np.max(train_shape)**3

        # volume test set
        test_shape = np.asarray(shape)
        test_shape[d] = split
        test_volume = np.prod(test_shape)
        test_cubic_hull_volume = np.max(test_shape)**3

        # maximize covered fraction
        covered_fraction = train_volume / train_cubic_hull_volume + test_volume / test_cubic_hull_volume
        if covered_fraction > maximum_covered_fraction:
            maximum_covered_fraction = covered_fraction
            best_dim = d

    return best_dim, splits[best_dim]


def train_test_split(x, y=None, test_size=0.33):
    """
    Splits a (both labeled and unlabeled) data in a stratified train and test set while maximizing isotropic data dimensions

    :param x: input data
    :param y: corresponding labels, optional
    :param test_size: relative size of the test set
    :return: train test splits of the data
    """

    if y is None:  # unlabeled dataset, easier case
        splits = [int(sz * (1-test_size)) for sz in x.shape]
        d, s = _maximize_isotropy(x.shape, splits)
        x_train, x_test = np.split(x, [s], d)
        return x_train, x_test
    else:  # labeled dataset, additionally make sure labels are balanced in train/test folds
        splits = [_find_split(y, test_size=test_size, axis=axis) for axis in [0, 1, 2]]
        d, s = _maximize_isotropy(y.shape, splits)
        x_train, x_test = np.split(x, [s], d)
        y_train, y_test = np.split(y, [s], d)
        return x_train, y_train, x_test, y_test

