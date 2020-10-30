import numpy as np
from scipy.spatial.distance import directed_hausdorff


def _jaccard_part_metrics(y_true, y_pred, w=None, max_elements=1000000):
    """
    computes the confusion matrix metrics in a memory efficient way

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :param max_elements: maximum number of elements for block based computing
    :return: the Jaccard index, accuracy, balanced accuracy, precision, recall and f1-score
    """

    # flatten, apply masking if necessary and binarize
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if w is not None:
        w = w.flatten()
        y_true = y_true[w]
        y_pred = y_pred[w]
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5

    # compute metrics
    intersection = 0
    true_total = 0
    pred_total = 0
    n_blocks = y_true.size // max_elements
    for n in range(n_blocks + 1):
        # find start and stopping point
        start = max_elements * n
        stop = np.minimum(max_elements * (n+1), y_true.size)

        # block to process
        y_true_block = y_true[start:stop]
        y_pred_block = y_pred[start:stop]

        # processing
        intersection += np.sum(y_true_block * y_pred_block)
        true_total += np.sum(y_true_block)
        pred_total += np.sum(y_pred_block)

    return intersection, true_total, pred_total


def jaccard(y_true, y_pred, w=None):
    """
    Jaccard index between two segmentations

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the Jaccard index
    """

    # compute jaccard score
    intersection, true_total, pred_total = _jaccard_part_metrics(y_true, y_pred, w=w)
    union = true_total + pred_total - intersection

    return (intersection + 1) / (union + 1)


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


def _conf_matrix_metrics(y_true, y_pred, w=None, max_elements=1000000):
    """
    computes the confusion matrix metrics in a memory efficient way

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :param max_elements: maximum number of elements for block based computing
    :return: the Jaccard index, accuracy, balanced accuracy, precision, recall and f1-score
    """

    # flatten, apply masking if necessary and binarize
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if w is not None:
        w = w.flatten()
        y_true = y_true[w]
        y_pred = y_pred[w]
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5

    # compute metrics
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n_blocks = y_true.size // max_elements
    for n in range(n_blocks):
        # find start and stopping point
        start = max_elements * n
        stop = np.minimum(max_elements * (n+1), y_true.size)

        # block to process
        y_true_block = y_true[start:stop]
        y_pred_block = y_pred[start:stop]

        # processing
        tp += np.sum(y_true_block * y_pred_block)
        tn += np.sum((1 - y_true_block) * (1 - y_pred_block))
        fp += np.sum((1 - y_true_block) * y_pred_block)
        fn += np.sum(y_true_block * (1 - y_pred_block))

    return tp, tn, fp, fn


def accuracy_metrics(y_true, y_pred, w=None):
    """
    Accuracy metrics between two segmentations (accuracy, balanced accuracy, precision, recall and f1-score)

    :param y_true: (N1, N2, ...) array of the true labels
    :param y_pred: (N1, N2, ...) array of the predictions (either probs or binary)
    :param w: (N1, N2, ...) masking array
    :return: the accuracy, balanced accuracy, precision, recall and f1-score
    """

    # compute accuracy metrics
    tp, tn, fp, fn = _conf_matrix_metrics(y_true, y_pred, w=w)
    total = tp + tn + fp + fn
    accuracy = (tp + tn + 1) / (total + 1)
    recall = (tp + 1) / (tp + fn + 1)
    specificity = (tn + 1) / (tn + fp + 1)
    balanced_accuracy = (recall + specificity) / 2
    precision = (tp + 1) / (tp + fp + 1)
    f1 = 2 * (precision * recall) / (precision + recall)

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
