import cv2
import numpy as np
import numpy.random as rnd
import torch
import torch.nn.functional as F
from scipy.ndimage import spline_filter1d


class ToTensor(object):

    def __init__(self, cuda=True):
        """
        Transforms a numpy array into a tensor
        :param cuda: specifies whether the tensor should be transfered to the GPU or not
        """
        self.cuda = cuda

    def __call__(self, x):
        """
        Forward call
        :param x: input array (N_1, N_2, N_3, ...)
        :return: output tensor (N_1, N_2, N_3, ...)
        """
        if self.cuda:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)


class ToFloatTensor(object):

    def __init__(self, cuda=True):
        """
        Transforms a Tensor to a FloatTensor
        :param cuda: specifies whether the tensor should be transfered to the GPU or not
        """
        self.cuda = cuda

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (N_1, N_2, N_3, ...)
        :return: output tensor (N_1, N_2, N_3, ...)
        """
        if self.cuda:
            return x.float().cuda()
        else:
            return x.float()


class ToLongTensor(object):

    def __init__(self, cuda=True):
        """
        Transforms a Tensor to a LongTensor
        :param cuda: specifies whether the tensor should be transfered to the GPU or not
        """
        self.cuda = cuda

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (N_1, N_2, N_3, ...)
        :return: output tensor (N_1, N_2, N_3, ...)
        """
        if self.cuda:
            return x.long().cuda()
        else:
            return x.long()


class AddChannelAxis(object):

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (N_1, N_2, N_3, ...)
        :return: output tensor (1, N_1, N_2, N_3, ...)
        """
        return x.unsqueeze(0)


class AddNoise(object):

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=1.0, include_segmentation=False,
                 include_weak_segmentation=False):
        """
        Adds noise to the input
        :param prob: probability of adding noise
        :param sigma_min: minimum noise standard deviation
        :param sigma_max: maximum noise standard deviation
        :param include_segmentation: 2nd half of the batch will not be augmented as this is assumed to be a (partial) segmentation
        """
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.include_segmentation = include_segmentation
        self.include_weak_segmentation = include_weak_segmentation

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, N_1, N_2, ...)
        :return: output tensor (B, N_1, N_2, ...)
        """

        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            if self.include_segmentation:
                sz = np.asarray(x.size())
                sz[0] = sz[0] // 2
                sz = tuple(sz)
                noise = torch.cat((torch.normal(0, sigma, sz), torch.zeros(sz)), dim=0)
            else:
                noise = torch.normal(0, sigma, x.size())
            if x.is_cuda:
                noise = noise.cuda()
            return x + noise
        else:
            return x


class Normalize(object):

    def __init__(self, mu=0, std=1):
        """
        Normalizes the input
        :param mu: mean of the normalization
        :param std: standard deviation of the normalization
        """
        self.mu = mu
        self.std = std

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (N_1, N_2, N_3, ...)
        :return: output tensor (N_1, N_2, N_3, ...)
        """
        return (x - self.mu) / self.std


class Scale(object):

    def __init__(self, scale_factor=(0.5, 1.5), mode='bilinear'):
        """
        Scales the input by a specific factor (randomly selected from a minimum-maximum range)
        :param scale_factor: minimum and maximum scaling factor
        """
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, [Z  , Y  ,] X)
        :return: output tensor (B, C, [Z' , Y' ,] X)
        """
        if type(self.scale_factor) == tuple:
            scale_factor = (self.scale_factor[1] - self.scale_factor[0]) * np.random.random_sample() + \
                           self.scale_factor[0]
        else:
            scale_factor = self.scale_factor
        return F.interpolate(x, scale_factor=scale_factor, mode=self.mode, align_corners=False)


class RandomCrop_2D(object):

    def __init__(self, crop_shape):
        """
        Selects a random crop from the input
        :param crop_shape: 2D shape of the crop
        """
        self.crop_shape = crop_shape

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y', X')
        """
        r = np.random.randint(0, x.size(2) - self.crop_shape[0] + 1)
        c = np.random.randint(0, x.size(3) - self.crop_shape[1] + 1)
        return x[:, :, r:r + self.crop_shape[0], c:c + self.crop_shape[1]]


class RandomCrop_3D(object):

    def __init__(self, crop_shape):
        """
        Selects a random crop from the input
        :param crop_shape: 3D shape of the crop
        """
        self.crop_shape = crop_shape

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z , Y , X)
        :return: output tensor (B, C, Z', Y', X')
        """
        r = np.random.randint(0, x.size(2) - self.crop_shape[0] + 1)
        c = np.random.randint(0, x.size(3) - self.crop_shape[1] + 1)
        z = np.random.randint(0, x.size(4) - self.crop_shape[2] + 1)
        return x[:, :, r:r + self.crop_shape[0], c:c + self.crop_shape[1], z:z + self.crop_shape[2]]


class FlipX_2D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the X axis
        :param shape: 2D shape of the input image
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = tuple(shape)
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        xv, yv = np.meshgrid(i, j)
        xv = np.fliplr(xv).copy()

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class FlipX_3D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the X axis
        :param shape: 3D shape of the inputs
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        k = np.linspace(-1, 1, shape[2])
        xv, yv, zv = np.meshgrid(i, j, k)
        xv = np.fliplr(xv).copy()

        grid = torch.cat(
            (torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1), torch.Tensor(zv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z, Y , X)
        :return: output tensor (B, C, Z, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class FlipY_2D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the Y axis
        :param shape: 2D shape of the input input image
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = tuple(shape)
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        xv, yv = np.meshgrid(i, j)
        yv = np.flipud(yv).copy()

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class FlipY_3D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the Y axis
        :param shape: 3D shape of the inputs
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        k = np.linspace(-1, 1, shape[2])
        xv, yv, zv = np.meshgrid(i, j, k)
        yv = np.flipud(yv).copy()

        grid = torch.cat(
            (torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1), torch.Tensor(zv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z, Y , X)
        :return: output tensor (B, C, Z, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class Rotate90_2D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Rotate the inputs by 90 degree angles
        :param shape: 2D shape of the input input image
        :param prob: probability of rotating
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = tuple(shape)
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        grids = []
        for m in range(4):
            xv, yv = np.meshgrid(i, j)
            xv = np.rot90(xv, m + 1).copy()
            yv = np.rot90(yv, m + 1).copy()

            grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
            grid = grid.unsqueeze(0)
            if cuda:
                grid = grid.cuda()
            grids.append(grid)
        self.grids = grids

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grids[rnd.randint(0, 4)].repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class Rotate90_3D(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Rotate the inputs by 90 degree angles
        :param shape: 3D shape of the input input image
        :param prob: probability of rotating
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        k = np.linspace(-1, 1, shape[2])
        grids = []
        for m in range(4):
            xv, yv, zv = np.meshgrid(i, j, k)
            xv = np.rot90(xv, m + 1).copy()
            yv = np.rot90(yv, m + 1).copy()
            zv = np.rot90(zv, m + 1).copy()

            grid = torch.cat(
                (torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1), torch.Tensor(zv).unsqueeze(-1)),
                dim=-1)
            grid = grid.unsqueeze(0)
            if cuda:
                grid = grid.cuda()
            grids.append(grid)
        self.grids = grids

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z, Y , X)
        :return: output tensor (B, C, Z, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self.grids[rnd.randint(0, 4)].repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class RotateRandom_2D(object):

    def __init__(self, shape, prob=1.0, range_=200, cuda=True):
        """
        Rotate the inputs by a random amount of degrees within interval.
        :param shape: 2D shape of the input image
        :param range_: random degree interval size (symmetric around 0)
        :param cuda: specify whether the inputs are on the GPU.
        """
        self.shape = tuple(shape)
        self.cuda = cuda
        self.range = int(range_ / 2)
        self.prob = prob
        self.image_center = int(self.shape[0] / 2), int(self.shape[1] / 2)

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        self.xv, self.yv = np.meshgrid(i, j)

    def _rotation_grid(self):
        angle = np.random.randint(-self.range, self.range)
        rot_matrix = cv2.getRotationMatrix2D(self.image_center, angle, 1.0)
        xv = cv2.warpAffine(self.xv, rot_matrix, self.xv.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=2)
        yv = cv2.warpAffine(self.yv, rot_matrix, self.yv.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=2)

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if self.cuda:
            grid = grid.cuda()
        return grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self._rotation_grid()
            grid = grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class RotateRandom_3D(object):

    def __init__(self, shape, prob=1.0, range_=200, cuda=True):
        """
        Rotate the inputs by a random amount of degrees within interval.
        :param shape: 3D shape of the input image
        :param range_: random degree interval size (symmetric around 0)
        :param cuda: specify whether the inputs are on the GPU.
        """
        self.shape = tuple(shape)
        self.cuda = cuda
        self.range = int(range_ / 2)
        self.prob = prob
        self.image_center = int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        k = np.linspace(-1, 1, shape[2])
        self.xv, self.yv, self.zv = np.meshgrid(i, j, k)

    def _rotation_grid(self):
        angle = np.random.randint(-self.range, self.range)
        rot_matrix = cv2.getRotationMatrix2D(self.image_center, angle, 1.0)
        xv = cv2.warpAffine(self.xv, rot_matrix, self.xv.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=2)
        yv = cv2.warpAffine(self.yv, rot_matrix, self.yv.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=2)
        zv = cv2.warpAffine(self.zv, rot_matrix, self.zv.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=2)

        grid = torch.cat(
            (torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1), torch.Tensor(zv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if self.cuda:
            grid = grid.cuda()
        return grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z, Y , X)
        :return: output tensor (B, C, Z, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self._rotation_grid()
            grid = grid.repeat_interleave(x.size(0), dim=0)
            return F.grid_sample(x, grid)
        else:
            return x


class RandomDeformation_2D(object):

    def __init__(self, shape, prob=1, cuda=True, points=None, sigma=0.01, include_segmentation=False,
                 sampling_interval=64):
        """
        Apply random deformation to the inputs
        :param shape: 2D shape of the input image
        :param prob: probability of deforming the data
        :param cuda: specifies whether the inputs are on the GPU
        :param points: seed points for deformation
        :param sigma: standard deviation for deformation
        :param include_segmentation: 2nd half of the batch needs casting to integers because of warping
        :param sampling_interval: determine amount of points to be sampled for deformation
        """
        self.shape = tuple(shape)
        self.prob = prob
        self.cuda = cuda
        if points == None:
            points = [shape[0] // sampling_interval, shape[1] // sampling_interval]
        self.points = points
        self.sigma = sigma
        self.p = 10
        self.include_segmentation = include_segmentation

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        xv, yv = np.meshgrid(i, j)

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def _deformation_grid(self):
        """
        Get a new random deformation grid
        :return: deformation tensor
        """
        sigma = np.random.rand() * self.sigma
        displacement = np.random.randn(*self.points, 2) * sigma

        # filter the displacement
        displacement_f = np.zeros_like(displacement)
        for d in range(0, displacement.ndim - 1):
            spline_filter1d(displacement, axis=d, order=3, output=displacement_f, mode='nearest')
            displacement = displacement_f

        # resample to proper size
        displacement_f = np.zeros((self.shape[0], self.shape[1], 2))
        for d in range(0, displacement.ndim - 1):
            displacement_f[:, :, d] = cv2.resize(displacement[:, :, d], dsize=self.shape,
                                                 interpolation=cv2.INTER_CUBIC)

        displacement = torch.Tensor(displacement_f).unsqueeze(0)
        if self.cuda:
            displacement = displacement.cuda()
        grid = self.grid + displacement

        return grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Y , X)
        :return: output tensor (B, C, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self._deformation_grid()
            grid = grid.repeat_interleave(x.size(0), dim=0)
            x_aug = F.grid_sample(x, grid, padding_mode="border")
            if self.include_segmentation:
                x_aug[x.size(0) // 2:, ...] = torch.round(x_aug[x.size(0) // 2:, ...])
            return x_aug
        else:
            return x


class RandomDeformation_3D(object):

    def __init__(self, shape, prob=1, cuda=True, points=None, sigma=0.01, include_segmentation=False):
        """
        Apply random deformation to the inputs
        :param shape: shape of the inputs
        :param prob: probability of deforming the data
        :param cuda: specifies whether the inputs are on the GPU
        :param points: seed points for deformation
        :param sigma: standard deviation for deformation
        :param include_segmentation: 2nd half of the batch needs casting to integers because of warping
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda
        if points == None:
            points = [shape[0] // 64, shape[1] // 64, shape[2] // 32]
        self.points = points
        self.sigma = sigma
        self.p = 10
        self.include_segmentation = include_segmentation

        i = np.linspace(-1, 1, shape[0])
        j = np.linspace(-1, 1, shape[1])
        k = np.linspace(-1, 1, shape[2])
        xv, yv, zv = np.meshgrid(i, j, k)

        grid = torch.cat(
            (torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1), torch.Tensor(zv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def _deformation_grid(self):
        """
        Get a new random deformation grid
        :return: deformation tensor
        """
        sigma = np.random.rand() * self.sigma
        displacement = np.random.randn(*self.points, 2) * sigma

        # filter the displacement
        displacement_f = np.zeros_like(displacement)
        for d in range(0, displacement.ndim - 1):
            spline_filter1d(displacement, axis=d, order=3, output=displacement_f, mode='nearest')
            displacement = displacement_f

        # resample to proper size
        displacement_f = np.zeros((self.shape[0], self.shape[1], self.shape[2], 2))
        for d in range(0, displacement.ndim - 1):
            displacement_f[:, :, :, d] = cv2.resize(displacement[:, :, :, d], dsize=self.shape,
                                                    interpolation=cv2.INTER_CUBIC)

        displacement = torch.Tensor(displacement_f).unsqueeze(0)
        if self.cuda:
            displacement = displacement.cuda()
        grid = self.grid + displacement

        return grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor (B, C, Z, Y , X)
        :return: output tensor (B, C, Z, Y , X)
        """

        if rnd.rand() < self.prob:
            grid = self._deformation_grid()
            grid = grid.repeat_interleave(x.size(0), dim=0)
            x_aug = F.grid_sample(x, grid, padding_mode="border")
            if self.include_segmentation:
                x_aug[x.size(0) // 2:, ...] = torch.round(x_aug[x.size(0) // 2:, ...])
            return x_aug
        else:
            return x
