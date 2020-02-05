import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, balanced_accuracy_score, jaccard_score, precision_score, recall_score, \
    f1_score


def jaccard(y_true, y_pred, w=None):
    """
    Jaccard index between two segmentations
    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    # check mask
    if w is None:
        w = np.ones_like(y_true, dtype='bool')
    y_true = y_true[w]
    y_pred = y_pred[w]

    # binarize
    y_true = (y_true > 0.5).astype('int')
    y_pred = (y_pred > 0.5).astype('int')

    return jaccard_score(y_true, y_pred)


def dice(y_true, y_pred, w=None):
    """
    Dice coefficient between two segmentations
    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    j = jaccard(y_true, y_pred, w=w)

    return 2 * j / (1 + j)


def accuracy_metrics(y_true, y_pred, w=None):
    """
    Accuracy metrics between two segmentations:
        - Accuracy
        - Balanced accuracy
        - Precision
        - Recall
        - F1-score
    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    # check mask
    if w is None:
        w = np.ones_like(y_true, dtype='bool')
    y_true = y_true[w]
    y_pred = y_pred[w]

    # binarize
    y_true = (y_true > 0.5).astype('int')
    y_pred = (y_pred > 0.5).astype('int')

    # compute accuracy metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, balanced_accuracy, precision, recall, f1


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
