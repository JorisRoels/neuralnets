import augmentation.augmentation_2d as aug2d
import augmentation.augmentation_3d as aug3d
import torch
from torchvision.transforms import Compose


def get_augmenter2d(crop_shape, sample_shape=None, scale_factor=(0.75, 1.25), deformation_scale=0.01, sigma_max=0.1,
                    include_segmentation=False, tofloat=True, rotate90=True, rotate_rand=False, scale=True, crop=False,
                    flipx=True, flipy=True, rand_deform=True, noise=True):
    """
    Default augmenter for 2D data
    :param crop_shape: shape of the cropped inputs
    :param sample_shape: shape of the inputs before cropping
    :param scale_factor: minimum and maximum rescaling factors (or a constant factor)
    :param deformation_scale: scale of the deformations
    :param sigma_max: maximum noise standard deviation
    :param include_segmentation: flag that specifies whether the second half of the batch are segmentations
    :param tofloat: transform to float tensors
    :param rotate90: include 90 degree rotations
    :param rotate_rand: include random rotations
    :param scale: include scaling
    :param crop: include random crops
    :param flipx: include horizontal flips
    :param flipy: include vertical flips
    :param rand_deform: include random deformations
    :param noise: include noise augmentations
    :return: the augmenter
    """
    if sample_shape is None:
        sample_shape = [crop_shape[0] * 2, crop_shape[1] * 2]
    augmenter = Compose([])
    if tofloat:
        augmenter = Compose([augmenter, aug2d.ToFloatTensor(cuda=torch.cuda.is_available())])
    if rotate90:
        augmenter = Compose([augmenter, aug2d.Rotate90(sample_shape)])
    if rotate_rand:
        augmenter = Compose([augmenter, aug2d.RotateRandom(sample_shape)])
    if scale:
        augmenter = Compose([augmenter, aug2d.Scale(scale_factor=scale_factor)])
    if crop:
        augmenter = Compose([augmenter, aug2d.RandomCrop(crop_shape=crop_shape)])
    if flipx:
        augmenter = Compose([augmenter, aug2d.FlipX(crop_shape, prob=0.5)])
    if flipy:
        augmenter = Compose([augmenter, aug2d.FlipY(crop_shape, prob=0.5)])
    if rand_deform:
        augmenter = Compose([augmenter, aug2d.RandomDeformation(crop_shape, sigma=deformation_scale,
                                                                include_segmentation=include_segmentation)])
    if noise:
        augmenter = Compose([augmenter, aug2d.AddNoise(sigma_max=sigma_max, include_segmentation=include_segmentation)])

    return augmenter


def get_augmenter3d(crop_shape, sample_shape=None, scale_factor=(0.75, 1.25), deformation_scale=0.01, sigma_max=0.1,
                    include_segmentation=False, tofloat=True, rotate90=True, rotate_rand=False, scale=True, crop=False,
                    flipx=True, flipy=True, rand_deform=True, noise=True):
    """
    Default augmenter for 3D data
    :param crop_shape: shape of the cropped inputs
    :param sample_shape: shape of the inputs before cropping
    :param scale_factor: minimum and maximum rescaling factors (or a constant factor)
    :param deformation_scale: scale of the deformations
    :param sigma_max: maximum noise standard deviation
    :param include_segmentation: flag that specifies whether the second half of the batch are segmentations
    :param tofloat: transform to float tensors
    :param rotate90: include 90 degree rotations
    :param rotate_rand: include random rotations
    :param scale: include scaling
    :param crop: include random crops
    :param flipx: include horizontal flips
    :param flipy: include vertical flips
    :param rand_deform: include random deformations
    :param noise: include noise augmentations
    :return: the augmenter
    """
    if sample_shape is None:
        sample_shape = [crop_shape[0] * 2, crop_shape[1] * 2, crop_shape[2] * 2]
    augmenter = Compose([])
    if tofloat:
        augmenter = Compose([augmenter, aug3d.ToFloatTensor(cuda=torch.cuda.is_available())])
    if rotate90:
        augmenter = Compose([augmenter, aug3d.Rotate90(sample_shape)])
    if rotate_rand:
        augmenter = Compose([augmenter, aug2d.RotateRandom(sample_shape)])
    if scale:
        augmenter = Compose([augmenter, aug3d.Scale(scale_factor=scale_factor)])
    if crop:
        augmenter = Compose([augmenter, aug3d.RandomCrop(crop_shape=crop_shape)])
    if flipx:
        augmenter = Compose([augmenter, aug3d.FlipX(crop_shape, prob=0.5)])
    if flipy:
        augmenter = Compose([augmenter, aug3d.FlipY(crop_shape, prob=0.5)])
    if rand_deform:
        augmenter = Compose([augmenter, aug3d.RandomDeformation(crop_shape, sigma=deformation_scale,
                                                                include_segmentation=include_segmentation)])
    if noise:
        augmenter = Compose([augmenter, aug3d.AddNoise(sigma_max=sigma_max, include_segmentation=include_segmentation)])

    return augmenter
