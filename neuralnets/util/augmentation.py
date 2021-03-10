import numpy as np
import numpy.random as rnd
import torch
from torchvision.transforms import Compose
from elasticdeform import deform_random_grid
from skimage.transform import resize, rotate


def split_segmentation_transforms(transforms):
    """
    Splits the transforms in a shared part, and one specifically for the inputs and outputs

    :param transforms: transform object (Compose)
    :return: subsets of transforms (Compose): shared_transforms, x_transforms and y_transforms
    """

    shared_transforms = []
    x_transforms = []
    y_transforms = []
    for transform in transforms.transforms:
        if transform.__class__ == AddNoise or transform.__class__ == Normalize or \
                transform.__class__ == ContrastAdjust:
            x_transforms.append(transform)
        elif transform.__class__ == CleanDeformedLabels:
            y_transforms.append(transform)
        else:
            shared_transforms.append(transform)
    return Compose(shared_transforms), Compose(x_transforms), Compose(y_transforms)


class ToTensor(object):
    """
    Transforms a numpy array into a tensor

    :param forward x: input array (N_1, N_2, N_3, ...)
    :return: output tensor (N_1, N_2, N_3, ...)
    """

    def __call__(self, x):
        return torch.from_numpy(x)


class ToFloat(object):
    """
    Transforms a Tensor to a FloatTensor

    :param forward x: input array (N_1, N_2, N_3, ...)
    :return: output tensor (N_1, N_2, N_3, ...)
    """

    def __call__(self, x):
        return x.float()


class ToLong(object):
    """
    Transforms a Tensor to a LongTensor

    :param forward x: input array (N_1, N_2, N_3, ...)
    :return: output tensor (N_1, N_2, N_3, ...)
    """

    def __call__(self, x):
        return x.long(),


class AddNoise(object):
    """
    Adds noise to the input

    :param initialization prob: probability of adding noise
    :param initialization sigma_min: minimum noise standard deviation
    :param initialization sigma_max: maximum noise standard deviation
    :param forward x: input array (N_1, N_2, ...)
    :return: output array (N_1, N_2, ...)
    """

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=1.0):
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):

        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            noise = rnd.randn(*x.shape) * sigma
            return x + noise
        else:
            return x


class Normalize(object):
    """
    Normalizes the input

    :param type: the desired type of normalization (z, unit or minmax)
    :param optional bits: number of bits used to represent a pixel value (only if type is unit)
    :param optional mu: normalization mean (only if type is z)
    :param optional sigma: normalization std (only if type is z)
    :param forward x: input array (N_1, N_2, ...)
    :return: output array (N_1, N_2, N_3, ...)
    """

    def __init__(self, type='unit', bits=8, mu=None, sigma=None):
        self.type = type
        self.bits = bits
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        if self.type == 'minmax':
            # apply minmax normalization
            m = np.min(x)
            M = np.max(x)
            eps = 1e-5
            return (x - m + eps) / (M - m + eps)
        elif self.type == 'unit':
            # apply unit normalization
            return x / (2**self.bits)
        else:
            # apply z normalization
            mu = np.mean(x) if self.mu is None else self.mu
            sigma = np.std(x) if self.sigma is None else self.sigma
            return (x - mu) / sigma


class ContrastAdjust(object):
    """
    Apply contrast adjustments to the data

    :param initialization prob: probability of adjusting contrast
    :param initialization adj: maximum adjustment (maximum intensity shift for minimum and maximum the new histogram)
    :param forward x: input array (N_1, N_2, N_3, ...)
    :return: output array (N_1, N_2, N_3, ...)
    """

    def __init__(self, prob=1, adj=0.1):
        self.prob = prob
        self.adj = adj

    def __call__(self, x):

        if rnd.rand() < self.prob:
            x_ = x

            m = np.min(x_)
            M = np.max(x_)
            r1 = rnd.rand()
            r2 = rnd.rand()
            m_ = 2 * self.adj * r1 - self.adj + m
            M_ = 2 * self.adj * r2 - self.adj + M

            if m != M:
                return ((x - m) / (M - m)) * (M_ - m_) + m_
            else:
                return x
        else:
            return x


class Scale(object):
    """
    Scales the input by a specific factor (randomly selected from a minimum-maximum range)

    :param initialization dsize: target size [Z' ,] Y' , X'
    :param initialization mode: padding type (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’)
    :param forward x: input array (C, [Z  ,] Y  , X)
    :return: output tensor (C, [Z' ,] Y' , X')
    """

    def __init__(self, dsize, mode='edge'):
        self.dsize = dsize
        self.mode = mode

    def __call__(self, x):
        x_ = np.zeros_like(x)
        for c in range(x.shape[0]):
            x_[c] = resize(x[c], self.dsize, mode=self.mode)
        return x_


class Flip(object):
    """
    Perform a flip along a specific dimension

    :param initialization prob: probability of flipping
    :param initialization dim: (spatial) dimension to flip (excluding the channel dimension, default flips last dim)
    :param forward x: input array (C, N_1, N_2, ...)
    :return: output array (C, N_1, N_2, ...)
    """

    def __init__(self, prob=0.5, dim=None):
        self.prob = prob
        if dim is None:
            self.dim = -1
        else:
            self.dim = dim + 1

    def __call__(self, x):

        if rnd.rand() < self.prob:
            return np.flip(x, axis=self.dim).copy()
        else:
            return x


class Rotate90(object):
    """
    Rotate the inputs by 90 degree angles

    :param initialization prob: probability of rotating
    :param forward x: input array (C, N_1, N_2, ...)
    :return: output array (C, N_1, N_2, ...)
    """

    def __init__(self, prob=1):
        self.prob = prob

    def __call__(self, x):

        if rnd.rand() < self.prob:
            n = x.ndim
            return np.rot90(x, k=rnd.randint(0, 4), axes=[n - 2, n - 1]).copy()
        else:
            return x


class RotateRandom(object):
    """
    Rotate the inputs by a random amount of degrees within interval.

    :param initialization angle: max rotation angle (sampled symmetrically around 0)
    :param initialization resize: resize to fit
    :param initialization mode: padding type (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’)
    :param forward x: input array (C, [Z ,] Y , X)
    :return: output array (C, [Z ,] Y , X)
    """

    def __init__(self, prob=1.0, angle=30, resize=False, mode='edge'):
        self.prob = prob
        self.angle = angle
        self.resize = resize
        self.mode = mode

    def __call__(self, x):

        if rnd.rand() < self.prob:
            angle = - self.angle + rnd.rand() * 2 * self.angle
            x_ = np.zeros_like(x)
            for c in range(x.shape[0]):
                x_[c] = rotate(x[c], angle, resize=self.resize, mode=self.mode)
            return x_
        else:
            return x


class RandomCrop(object):
    """
    Selects a random crop from the input

    :param initialization crop_shape: shape of the crop [Z' ,] Y' , X'
    :param forward x: input array (C, [Z ,] Y , X)
    :return: output tensor (C, [Z' ,] Y' , X')
    """

    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, x):
        z_ = np.random.randint(0, x.size(1) - self.crop_shape[0] + 1)
        y_ = np.random.randint(0, x.size(2) - self.crop_shape[1] + 1)
        if len(self.crop_shape) == 2:  # 2D
            return x[:, z_:z_ + self.crop_shape[0], y_:y_ + self.crop_shape[1]]
        else:  # 3D
            x_ = np.random.randint(0, x.size(3) - self.crop_shape[2] + 1)
            return x[:, z_:z_ + self.crop_shape[0], y_:y_ + self.crop_shape[1], x_:x_ + self.crop_shape[2]]


class RandomDeformation(object):
    """
    Apply random deformation to the inputs

    :param initialization prob: probability of deforming the data
    :param initialization sigma: standard deviation of the normal deformation distribution
    :param initialization points: number of points of the deformation grid
    :param initialization mode: padding type (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’)
    :param forward x: input array (C, [Z ,] Y , X)
    :return: output tensor (C, [Z ,] Y , X)
    """

    def __init__(self, prob=1, sigma=10, points=5, mode='nearest'):
        self.prob = prob
        self.sigma = sigma
        self.points = points
        self.mode = mode

    def __call__(self, x):

        if rnd.rand() < self.prob:
            if x.ndim == 3:  # 2D
                return deform_random_grid(x, sigma=self.sigma, points=self.points, axis=(1, 2), mode=self.mode)
            else:  # 3D
                return deform_random_grid(x, sigma=self.sigma, points=self.points, axis=(1, 2, 3), mode=self.mode)
        else:
            return x


class CleanDeformedLabels(object):
    """
    Clean the deformed labels by mapping the floating point values to nearest integers

    :param initialization coi: classes of interest
    :param forward x: input array (C, [Z ,] Y , X)
    :return: output tensor (C, [Z ,] Y , X)
    """

    def __init__(self, coi):
        self.coi = coi

    def __call__(self, x):

        d = np.zeros((len(self.coi), *x.shape), dtype=x.dtype)
        for i, c in enumerate(self.coi):
            d[i] = np.abs(x - c)
        d = np.argmin(d, axis=0)

        for i, c in enumerate(self.coi):
            x[d == i] = c

        return x
