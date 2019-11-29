from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt
from skimage import measure


class CrossEntropyLoss(nn.Module):

    def __init__(self, class_weight=None, size_average=True):
        """
        Initialization of the cross entropy loss function
        :param class_weight: weights for the classes
        :param size_average: flag that specifies whether to apply size averaging at the end or not
        """

        super(CrossEntropyLoss, self).__init__()

        self.class_weight = class_weight
        self.size_average = size_average

    def forward(self, input, target, weight=None):

        # apply log softmax
        log_p = F.log_softmax(input, dim=1)

        # channels on the last axis
        input_size = input.size()
        for d in range(1, len(input_size) - 1):
            log_p = log_p.transpose(d, d + 1)
        log_p = log_p.contiguous()

        # reshape everything
        log_p = log_p[target[:, 0, ...].unsqueeze(-1).repeat_interleave(input_size[1], dim=-1) >= 0]
        log_p = log_p.view(-1, input_size[1])
        mask = target >= 0
        target = target[mask]

        # compute negative log likelihood
        loss = F.nll_loss(log_p, target, reduction='none', weight=self.class_weight)
        if weight is not None:
            weight = weight[mask]
            loss = weight * loss

        # size averaging if necessary
        if self.size_average:
            loss = loss.mean()

        return loss


class LpLoss(nn.Module):

    def __init__(self, p=2, size_average=True):
        """
        Initialization of the Lp loss function
        :param p: parameter for the loss function
        :param size_average: normalize sums to means
        """

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

    def __init__(self, size_average=True):
        """
        Initialization of the Lp loss function
        :param size_average: normalize sums to means
        """

        super(L2Loss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sqrt(torch.sum(torch.pow(target - target_rec, 2)))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class L1Loss(nn.Module):

    def __init__(self, size_average=True):
        """
        Initialization of the Lp loss function
        :param size_average: normalize sums to means
        """

        super(L1Loss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sum(torch.abs(target - target_rec))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class MSELoss(nn.Module):

    def __init__(self, size_average=True):
        """
        Initialization of the Lp loss function
        :param size_average: normalize sums to means
        """

        super(MSELoss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sum(torch.pow(target - target_rec, 2))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class MADLoss(nn.Module):

    def __init__(self, size_average=True):
        """
        Initialization of the Lp loss function
        :param size_average: normalize sums to means
        """

        super(MADLoss, self).__init__()

        self.size_average = size_average

    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.sum(torch.abs(target - target_rec))
        if self.size_average:
            loss = loss / target.numel()
        return loss


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):

        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = 1 + logvar - mu.pow(2) - logvar.exp()
        kld = torch.mean(-0.5 * torch.sum(klds, dim=1), dim=0)

        return kld


def boundary_weight_map(labels, sigma=20, w0=1):
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
