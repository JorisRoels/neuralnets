import copy
import random

import numpy as np
import torch
from skimage.color import label2rgb


def sample_labeled_input(data, labels, input_shape):
    """
    Generate an input and target sample of certain shape from a labeled dataset
    :param data: data to sample from
    :param labels: labels to sample from
    :param input_shape: shape of the sample
    :return: a random sample
    """
    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
    y = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
    z = np.random.randint(0, data.shape[2] - input_shape[2] + 1)

    # extract input and target patch
    input = data[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]
    target = labels[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]

    return copy.copy(input), copy.copy(target)


def sample_unlabeled_input(data, input_shape):
    """
    Generate an input and target sample of certain shape from an unlabeled dataset
    :param data: data to sample from
    :param input_shape: shape of the sample
    :return: a random sample
    """

    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0] - input_shape[0] + 1)
    y = np.random.randint(0, data.shape[1] - input_shape[1] + 1)
    z = np.random.randint(0, data.shape[2] - input_shape[2] + 1)

    # extract input and target patch
    input = data[x:x + input_shape[0], y:y + input_shape[1], z:z + input_shape[2]]

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


def overlay(x, y, alpha=0.3, colors=[[0, 1, 0]], bg_label=0):
    """
    Overlay an image with a segmentation map
    :param x: input (grayscale) image
    :param y: label map (all zeros are assumed background)
    :param alpha: opacity level of the overlay
    :param colors: list of colors
    :param bg_label: background label
    :return: a color image with the labels overlayed
    """
    return label2rgb(y, image=x, alpha=alpha, colors=colors, bg_label=bg_label)


def set_seed(seed):
    """
    Sets the seed of all randomized modules (useful for reproducibility)
    :param seed: seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
