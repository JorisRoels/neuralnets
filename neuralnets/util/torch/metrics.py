import torch


def iou(y_true, y_pred, w=None):
    """
    Intersection-over-union metric

    :param y_true: (N1, N2, ...) tensor of the true labels
    :param y_pred: (N1, N2, ...) tensor of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking tensor
    :return: the IoU
    """

    # threshold
    y_pred = y_pred > 0.5

    # mask if necessary
    if w is not None:
        y_pred = y_pred[w]
        y_true = y_true[w]

    # compute jaccard score
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    return (intersection + 1) / (union + 1)


def dice(y_true, y_pred, w=None):
    """
    Dice coefficient between two segmentations

    :param y_true: (N1, N2, ...) tensor of the true labels
    :param y_pred: (N1, N2, ...) tensor of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking tensor
    :return: the Dice coefficient
    """

    j = iou(y_true, y_pred, w=w)

    return 2 * j / (1 + j)
