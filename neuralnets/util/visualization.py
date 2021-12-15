import numpy as np
from skimage.measure import label
from skimage.segmentation import mark_boundaries
import cv2


COLORS = np.zeros((6, 3))
COLORS[0] = (1, 0, 0)
COLORS[1] = (0, 1, 0)
COLORS[2] = (0, 0, 1)
COLORS[3] = (1, 1, 0)
COLORS[4] = (0, 1, 1)
COLORS[5] = (1, 0, 1)


def overlay(x, y, colors=None, alpha=0.5, boundaries=False):
    """
    Overlay an image with a segmentation map

    :param x: input image (either [X, Y, 3], [X, Y, 1], or [X, Y])
    :param y: label map (all zeros are assumed background), same shape as input image
    :param colors: a list of 3-tuples specifying the colors that correspond to the different labels (randomly chosen if not specified)
    :param alpha: opacity of the label overlay
    :param boundaries: boolean that specifies whether a boundary map should be extracted prior to the overlay
    :return: a color image with the labels overlayed
    """
    y = y.astype('uint8')

    # preprocess
    if x.ndim == 3:
        if x.shape[2] == 1:
            x = x[..., 0]

    # if colors are not specific, generate random colors
    if colors is None:
        colors = []
        for l in range(np.max(y)):
            colors.append(tuple(np.random.rand(3)))

    # boundary case
    if boundaries:
        if x.ndim == 3 and x.shape[2] == 3:
            seg_img = x
        else:
            seg_img = np.stack((x, x, x), axis=2)
        for l in np.unique(y):
            if l != 255 and l != 0:
                seg_img = mark_boundaries(seg_img, label((y == l).astype('uint8')), mode='thick', color=colors[l])

    # segment case
    else:
        if x.ndim == 3 and x.shape[2] == 3:
            seg_img = x
            r = np.zeros_like(x[..., 0])
            g = np.zeros_like(x[..., 1])
            b = np.zeros_like(x[..., 2])
        else:
            seg_img = np.stack((x, x, x), axis=2)
            r = np.zeros_like(x)
            g = np.zeros_like(x)
            b = np.zeros_like(x)

        for l in np.unique(y):
            if l != 255 and l != 0:
                r[y == l] = colors[l][0]
                g[y == l] = colors[l][1]
                b[y == l] = colors[l][2]

        overlay = np.stack((r, g, b), axis=2)
        mask = overlay.sum(axis=2) == 0
        seg_img = np.maximum(np.minimum(cv2.addWeighted(overlay, alpha, seg_img, 1, -0.2), 1), 0)
        for d in range(3):
            seg_img_d = seg_img[..., d]
            if x.ndim == 3 and x.shape[2] == 3:
                seg_img_d[mask] = x[..., d][mask]
            else:
                seg_img_d[mask] = x[mask]
            seg_img[..., d] = seg_img_d

    return seg_img
