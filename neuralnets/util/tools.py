import copy
import random

import numpy as np
import torch

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


def load_net(model_file):
    """
    Load a pretrained pytorch network
    :param model_file: path to the checkpoint
    :return: a module that corresponds to the trained network
    """
    return torch.load(model_file)


def set_seed(seed):
    """
    Sets the seed of all randomized modules (useful for reproducibility)
    :param seed: seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
