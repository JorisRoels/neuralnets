from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt
from skimage import measure

from neuralnets.util.tools import tensor_to_device


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss function

    :param initalization class_weight: weights for the classes
    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :param forward weight: optional weight tensor (B, N_1, N_2, ...)
    :return: cross entropy loss
    """

    def __init__(self, class_weight=None, size_average=True):

        super(CrossEntropyLoss, self).__init__()

        self.class_weight = class_weight
        # normalize class weights if necessary
        if self.class_weight is not None:
            self.class_weight = torch.Tensor(self.class_weight / np.sum(self.class_weight))
        self.size_average = size_average

    def forward(self, logits, target, weight=None):

        # apply log softmax
        log_p = F.log_softmax(logits, dim=1)

        # channels on the last axis
        input_size = logits.size()
        for d in range(1, len(input_size) - 1):
            log_p = log_p.transpose(d, d + 1)
        log_p = log_p.contiguous()

        # reshape everything
        log_p = log_p[target[:, 0, ...].unsqueeze(-1).repeat_interleave(input_size[1], dim=-1) >= 0]
        log_p = log_p.view(-1, input_size[1])
        mask = target >= 0
        target = target[mask]

        # compute negative log likelihood
        if self.class_weight is not None:
            cw = tensor_to_device(self.class_weight, device=target.device.index)
        else:
            cw = None
        loss = F.nll_loss(log_p, target, reduction='none', weight=cw)
        if weight is not None:
            weight = weight[mask]
            loss = weight * loss

        # size averaging if necessary
        if self.size_average:
            loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal loss function (T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. Focal Loss for Dense Object Detection, 2017)

    :param initalization alpha: weights for the classes (C)
    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: focal loss
    """

    def __init__(self, gamma=2, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()

        self.alpha = alpha
        # normalize alpha if necessary
        if self.alpha is not None:
            self.alpha = torch.Tensor(self.alpha / np.sum(self.alpha))
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logits, target):

        # apply log softmax
        log_p = F.log_softmax(logits, dim=1)
        p = F.softmax(logits, dim=1)

        # channels on the last axis
        input_size = logits.size()
        for d in range(1, len(input_size) - 1):
            log_p = log_p.transpose(d, d + 1)
        log_p = log_p.contiguous()

        # reshape everything
        log_p = log_p[target[:, 0, ...].unsqueeze(-1).repeat_interleave(input_size[1], dim=-1) >= 0]
        log_p = log_p.view(-1, input_size[1])
        p = p.view(-1, input_size[1])
        mask = target >= 0
        target = target[mask]

        # compute negative log likelihood
        if self.alpha is not None:
            cw = tensor_to_device(self.alpha, device=target.device.index)
        else:
            cw = None
        loss = F.nll_loss((1 - p) ** self.gamma * log_p, target, reduction='none', weight=cw)

        # size averaging if necessary
        if self.size_average:
            loss = loss.mean()

        return loss


class DiceLoss(nn.Module):
    """
    Dice loss function

    :param initialization c: index of the class of index
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: dice loss
    """

    def __init__(self, c=1):
        super(DiceLoss, self).__init__()

        self.c = c

    def forward(self, logits, target):
        # apply softmax and select predictions of the class of interest
        p = F.softmax(logits, dim=1)[:, self.c:self.c + 1, ...]

        # dice loss
        numerator = 2 * torch.sum(p * target)
        denominator = torch.sum(p + target)

        return 1 - ((numerator + 1) / (denominator + 1))


class TverskyLoss(nn.Module):
    """
    Tversky loss function (S. S. M. Salehi, D. Erdogmus, and A. Gholipour. Tversky loss function for image segmentation using 3D fully convolutional deep networks, 2017)

    :param initialization c: index of the class of index
    :param forward logits: logits tensor (B, C, N_1, N_2, ...)
    :param forward target: targets tensor (B, N_1, N_2, ...)
    :return: tversky loss
    """

    def __init__(self, beta=0.5, c=1):
        super(TverskyLoss, self).__init__()

        self.beta = beta
        self.c = c

    def forward(self, logits, target):
        # apply softmax and select predictions of the class of interest
        p = F.softmax(logits, dim=1)[:, self.c:self.c + 1, ...]

        # tversky loss
        numerator = torch.sum(p * target)
        denominator = numerator + self.beta * torch.sum((1 - target) * p) + (1 - self.beta) * torch.sum(
            (1 - p) * target)

        return 1 - ((numerator + 1) / (denominator + 1))


class LpLoss(nn.Module):
    """
    L_p loss function

    :param initalization p: parameter for the loss function
    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: L_p loss
    """

    def __init__(self, p=2, size_average=True):
        super(LpLoss, self).__init__()

        self.p = p
        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.pow(torch.sum(torch.pow(torch.abs(target - target_rec), self.p)), 1 / self.p)
        if self.size_average:
            loss = loss / target.numel()
        return loss


class L2Loss(nn.Module):
    """
    L_2 loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward logits: logits tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: L_2 loss
    """

    def __init__(self, size_average=True):
        super(L2Loss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sqrt(torch.sum(torch.pow(target - target_rec, 2)))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class L1Loss(nn.Module):
    """
    L_1 loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward pred: predictions tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: L_1 loss
    """

    def __init__(self, size_average=True):
        super(L1Loss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = torch.sum(torch.abs(target - pred))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class MSELoss(nn.Module):
    """
    Mean squared error (MSE) loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward pred: predictions tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: MSE loss
    """

    def __init__(self, size_average=True):
        super(MSELoss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = torch.sum(torch.pow(target - pred, 2))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class MADLoss(nn.Module):
    """
    Mean absolute deviation (MAD) loss function

    :param initalization size_average: flag that specifies whether to apply size averaging at the end or not
    :param forward pred: predictions tensor (N_1, N_2, ...)
    :param forward target: targets tensor (N_1, N_2, ...)
    :return: MAD loss
    """

    def __init__(self, size_average=True):
        super(MADLoss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = torch.sum(torch.abs(target - pred))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class KLDLoss(nn.Module):
    """
    Kullback Leibler divergence (KLD) loss function

    :param forward mu: mean tensor (N_1, N_2, ...)
    :param forward log: logarithmic variance tensor (N_1, N_2, ...)
    :return: KLD loss
    """

    def forward(self, mu, logvar):

        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = 1 + logvar - mu.pow(2) - logvar.exp()
        kld = torch.mean(-0.5 * torch.sum(klds, dim=1), dim=0)

        return kld


def boundary_weight_map(labels, sigma=20, w0=1):
    """
    Compute the boundary weight map, according to the the original U-Net paper

    :param labels: input tensor
    :param optional sigma: damping parameter
    :param optional w0: initial value of the weight map
    :return boundary weight map as a tensor
    """

    y = labels.cpu().numpy()
    weight = np.ones(y.shape)
    diag = np.sqrt(weight.shape[2] ** 2 + weight.shape[3] ** 2)

    # for each image in the batch
    for b in range(y.shape[0]):

        # compute connected components
        comps = measure.label(y[b, 0, ...])
        n_comps = np.max(comps)

        if n_comps > 0:  # if there is at least one component
            if n_comps > 1:  # if there are at least two components
                # remove every component separately and compute distance transform
                dtfs = np.zeros((n_comps, comps.shape[0], comps.shape[1]))
                for c in range(n_comps):
                    # compute distance transform
                    y_ = copy(y[b, 0, ...])
                    y_[comps == (1 + c)] = 0  # remove component
                    dtfs[c, ...] = distance_transform_edt(1 - y_)
                dtfs_sorted = np.sort(dtfs, axis=0)
                dtf1 = dtfs_sorted[0, ...]
                dtf2 = dtfs_sorted[1, ...]
            else:
                # compute distance transform
                dtf1 = distance_transform_edt(1 - y[b, 0, ...])
                dtf2 = diag
            # update weight map
            weight[b, 0, ...] = weight[b, 0, ...] + w0 * (1 - y[b, 0, ...]) * np.exp(
                - (dtf1 + dtf2) ** 2 / (2 * sigma ** 2))

    return torch.Tensor(weight).cuda()


def _parse_loss_params(t):

    params = {}
    for s in t:
        key, val = s.split(':')
        if val.count(',') > 0:
            val = val.split(',')
            for i in range(len(val)):
                val[i] = float(val[i])
        else:
            val = float(val)
        params[key] = val

    return params


def get_loss_function(s):
    """
    Returns a loss function according to the settings in a string. The string is formatted as follows:
        s = <loss-function-name>[#<param>:<param-value>#<param>:<param-value>#...]
            parameter values are either
                - scalars
                - vectors (written as scalars, separated by commas)

    :param s: loss function specifier string, formatted as shown on top
    :return: the required loss function
    """
    t = s.lower().replace("-", "_").split('#')
    name = t[0]
    params = _parse_loss_params(t[1:])
    if name == "ce" or name == "cross_entropy":
        return CrossEntropyLoss(**params)
    elif name == "fl" or name == "focal":
        return FocalLoss(**params)
    elif name == "dl" or name == "dice":
        return DiceLoss(**params)
    elif name == "tl" or name == "tversky":
        return TverskyLoss(**params)
    elif name == "lp":
        return LpLoss(**params)
    elif name == "l2":
        return L2Loss(**params)
    elif name == "l1":
        return L1Loss(**params)
    elif name == "mse":
        return MSELoss(**params)
    elif name == "mad":
        return MADLoss(**params)
    elif name == "kld":
        return KLDLoss(**params)
