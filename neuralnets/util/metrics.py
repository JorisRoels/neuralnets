import numpy as np
from scipy.spatial.distance import directed_hausdorff


def jaccard(x, y):
    """
    Jaccard index between two segmentations
    :param x: array
    :param y: array
    :return: the Jaccard index
    """

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute jaccard
    intersection = np.sum(np.multiply(x, y))
    union = np.sum(x) + np.sum(y) - intersection
    return (intersection + eps) / (union + eps)


def dice(x, y):
    """
    Dice coefficient between two segmentations
    :param x: array
    :param y: array
    :return: the Dice coefficient
    """

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute dice
    intersection = np.sum(np.multiply(x, y))
    return 2 * (intersection + eps) / (np.sum(x) + np.sum(y) + eps)


def accuracy_metrics(x, y):
    """
    Accuracy, precision, recall and f-score between two segmentations
    :param x: array
    :param y: array
    :return: the accuracy metrics
    """

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    tp = np.sum(np.multiply(x, y))
    tn = np.sum(np.multiply(1 - x, 1 - y))
    fp = np.sum(np.multiply(x, 1 - y))
    fn = np.sum(np.multiply(1 - x, y))

    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f_score = (2 * (precision * recall) + eps) / (precision + recall + eps)

    return accuracy, precision, recall, f_score


def hausdorff_distance(x, y):
    """
    Hausdorff distance between two segmentations
    :param x: array
    :param y: array
    :return: the hausdorff distance
    """

    # binarize
    x = x > 0.5
    y = y > 0.5

    hd_0 = 0
    hd_1 = 0
    hd = 0
    for i in range(x.shape[0]):
        hd_0 += directed_hausdorff(x[i, ...], y[i, ...])[0]
        hd_1 += directed_hausdorff(y[i, ...], x[i, ...])[0]
        hd += max(hd_0, hd_1)
    hd_0 /= x.shape[0]
    hd_1 /= x.shape[0]
    hd /= x.shape[0]

    return hd, hd_0, hd_1
