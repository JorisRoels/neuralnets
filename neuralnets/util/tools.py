import copy
import random

import numpy as np
import torch
import torchvision.utils as vutils

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


def module_to_device(module, device):
    """
    Transfers a pytorch module to a specific GPU device

    :param module: module that should be transferred
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    """
    if not torch.cuda.is_available():
        module.cpu()
    else:
        module.cuda(device=device)


def tensor_to_device(x, device):
    """
    Transfers a pytorch tensor to a specific GPU device

    :param x: tensor or sequence/list of tensors that should be transferred
    :param device: index of the device (if there are no GPU devices, it will be moved to the CPU)
    :return x: same tensor, but switched to device
    """
    if isinstance(x, tuple) or isinstance(x, list):
        if not torch.cuda.is_available():
            return [xx.cpu() for xx in x]
        else:
            return [xx.cuda(device=torch.device('cuda:' + str(device))) for xx in x]
    else:
        if not torch.cuda.is_available():
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
