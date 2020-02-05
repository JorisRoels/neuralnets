import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar

from neuralnets.util.io import write_volume
from neuralnets.util.metrics import jaccard, accuracy_metrics, hausdorff_distance
from neuralnets.util.tools import gaussian_window


def sliding_window_multichannel(image, step_size, window_size, track_progress=False):
    """
    Iterator that acts as a sliding window over a multichannel 3D image
    :param image: multichannel image (4D array)
    :param step_size: step size of the sliding window (3-tuple)
    :param window_size: size of the window (3-tuple)
    :param track_progress: optionally, for tracking progress with progress bar
    """

    # define range
    zrange = [0]
    while zrange[-1] < image.shape[1] - window_size[0]:
        zrange.append(zrange[-1] + step_size[0])
    zrange[-1] = image.shape[1] - window_size[0]
    yrange = [0]
    while yrange[-1] < image.shape[2] - window_size[1]:
        yrange.append(yrange[-1] + step_size[1])
    yrange[-1] = image.shape[2] - window_size[1]
    xrange = [0]
    while xrange[-1] < image.shape[3] - window_size[2]:
        xrange.append(xrange[-1] + step_size[2])
    xrange[-1] = image.shape[3] - window_size[2]

    # loop over the range
    if track_progress:
        bar = Bar('Progress', max=len(zrange) * len(yrange) * len(xrange))
    for z in zrange:
        for y in yrange:
            for x in xrange:

                # yield the current window
                if window_size[0] == 1:  # 2D
                    yield (z, y, x, image[:, z, y:y + window_size[1], x:x + window_size[2]])
                else:  # 3D
                    yield (z, y, x, image[:, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]])
                if track_progress:
                    bar.next()
    if track_progress:
        bar.finish()


def segment_multichannel(data, net, input_shape, batch_size=1, step_size=None, train=False, track_progress=False):
    """
    Segment a multichannel 3D image using a specific network
    :param data: 4D array (C, Z, Y, X) representing the multichannel 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :return: the segmented image
    """

    # make sure we compute everything on the gpu and in the correct mode
    if torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()
    if train:
        net.train()
    else:
        net.eval()

    channels = data.shape[0]

    # 2D or 3D
    is2d = len(input_shape) == 2

    # set step size to half of the window if necessary
    if step_size == None:
        if is2d:
            step_size = (1, input_shape[0] // 2, input_shape[1] // 2)
        else:
            step_size = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2] // 2)

    # gaussian window for smooth block merging
    if is2d:
        g_window = gaussian_window((1, input_shape[0], input_shape[1]), sigma=input_shape[-1] / 4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)

    # allocate space
    seg_cum = np.zeros(data.shape[1:])
    counts_cum = np.zeros(data.shape[1:])

    # define sliding window
    if is2d:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=(1, input_shape[0], input_shape[1]),
                                         track_progress=track_progress)
    else:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=input_shape,
                                         track_progress=track_progress)

    # start prediction
    batch_counter = 0
    if is2d:
        batch = np.zeros((batch_size, channels, input_shape[0], input_shape[1]))
    else:
        batch = np.zeros((batch_size, channels, input_shape[0], input_shape[1], input_shape[2]))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:

            # convert to tensors
            if torch.cuda.is_available():
                inputs = torch.FloatTensor(batch).cuda()
            else:
                inputs = torch.FloatTensor(batch).cpu()

            # forward prop
            outputs = net(inputs)
            if type(outputs) is tuple:
                outputs = outputs[-1]
            outputs = F.softmax(outputs, dim=1)

            # cumulate segmentation volume
            for b in range(batch_size):
                (z_b, y_b, x_b) = positions[b, :]
                # take into account the gaussian filtering
                if is2d:
                    seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
                    counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
                else:
                    seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                        np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
                    counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

            # reset batch counter
            batch_counter = 0

    # don't forget last batch
    # convert to tensors
    if torch.cuda.is_available():
        inputs = torch.FloatTensor(batch).cuda()
    else:
        inputs = torch.FloatTensor(batch).cpu()

    # forward prop
    outputs = net(inputs)
    if type(outputs) is tuple:
        outputs = outputs[-1]
    outputs = F.softmax(outputs, dim=1)

    # cumulate segmentation volume
    for b in range(batch_counter):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            seg_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1:2, :, :])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            seg_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs.data.cpu().numpy()[b, 1, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window

    # crop out the symmetric extension and compute segmentation
    segmentation = np.divide(seg_cum, counts_cum)

    return segmentation


def segment(data, net, input_shape, batch_size=1, step_size=None, train=False, track_progress=False):
    """
    Segment a 3D image using a specific network
    :param data: 3D array (Z, Y, X) representing the 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :return: the segmented image
    """

    return segment_multichannel(data[np.newaxis, ...], net, input_shape,
                                batch_size=batch_size, step_size=step_size, train=train, track_progress=track_progress)


def validate(net, data, labels, input_size, label_of_interest=1, batch_size=1, write_dir=None, val_file=None,
             writer=None, epoch=0, track_progress=False):
    """
    Validate a network on a dataset and its labels
    :param net: image-to-image segmentation network
    :param data: 3D array (Z, Y, X) representing the 3D image
    :param labels: 3D array (Z, Y, X) representing the 3D labels
    :param input_size: size of the inputs (either 2 or 3-tuple) for processing
    :param label_of_interest: index of the label of interest
    :param batch_size: batch size for processing
    :param write_dir: optionally, specify a directory to write the output
    :param val_file: optionally, specify a file to write the validation results
    :param writer: optionally, summary writer for logging to tensorboard
    :param epoch: optionally, current epoch for logging to tensorboard
    :param track_progress: optionally, for tracking progress with progress bar
    :return: validation results, i.e. accuracy, precision, recall, f-score, jaccard and dice score
    """

    print('[%s] Validating the trained network' % (datetime.datetime.now()))

    if write_dir is not None and not os.path.exists(write_dir):
        os.mkdir(write_dir)

    segmentation = segment(data, net, input_size, batch_size=batch_size, track_progress=track_progress)
    labels_interest = (labels == label_of_interest).astype('float')
    j = jaccard(segmentation, labels_interest, w=labels != 255)
    a, ba, p, r, f = accuracy_metrics(segmentation, labels_interest, w=labels != 255)
    if np.sum(labels == 255) > 0:
        h = -1
    else:
        h = hausdorff_distance(segmentation, labels)[0]
    if write_dir is not None:
        print('[%s] Writing the output' % (datetime.datetime.now()))
        write_volume(255 * segmentation, write_dir, type='pngseq')
    if writer is not None:
        z = data.shape[0] // 2
        N = 1024
        if data.shape[1] > N:
            writer.add_image('val/input', data[z:z + 1, :N, :N], epoch)
            writer.add_image('val/segmentation', segmentation[z:z + 1, :N, :N], epoch)
        else:
            writer.add_image('val/input', data[z:z + 1, ...], epoch)
            writer.add_image('val/segmentation', segmentation[z:z + 1, ...], epoch)

    print('[%s] Network performance: Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
    if val_file is not None:
        np.save(val_file, np.asarray([a, p, r, f, j, d, h]))
    return a, ba, p, r, f, j, h
