import os

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar

from neuralnets.util.io import read_volume, write_volume, print_frm, mkdir
from neuralnets.util.numpy.metrics import iou, accuracy_metrics, hausdorff_distance, _iou_part_metrics, \
    _conf_matrix_metrics
from neuralnets.util.tools import gaussian_window, tensor_to_device, module_to_device, normalize


def sliding_window_multichannel(image, step_size, window_size, in_channels=1, track_progress=False,
                                normalization='unit'):
    """
    Iterator that acts as a sliding window over a multichannel 3D image

    :param image: multichannel image (4D array)
    :param step_size: step size of the sliding window (3-tuple)
    :param window_size: size of the window (3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param track_progress: optionally, for tracking progress with progress bar
    :param normalization: type of data normalization (unit, z or minmax)
    """

    # adjust z-channels if necessary
    window_size = np.asarray(window_size)
    is2d = window_size[0] == 1
    if is2d:  # 2D
        window_size[0] = in_channels

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
                if is2d:
                    input = image[0, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]]
                else:
                    input = image[:, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]]
                    yield (z, y, x, image[:, z:z + window_size[0], y:y + window_size[1], x:x + window_size[2]])
                input = normalize(input, type=normalization)
                yield (z, y, x, input)

                if track_progress:
                    bar.next()
    if track_progress:
        bar.finish()


def _init_step_size(step_size, input_shape, is2d):
    if step_size == None:
        if is2d:
            step_size = (1, max(1, input_shape[0] // 2), max(1, input_shape[1] // 2))
        else:
            step_size = (max(1, input_shape[0] // 2), max(1, input_shape[1] // 2), max(1, input_shape[2] // 2))
    return step_size


def _init_gaussian_window(input_shape, is2d):
    if is2d:
        g_window = gaussian_window((1, input_shape[0], input_shape[1]), sigma=input_shape[-1] / 4)
    else:
        g_window = gaussian_window(input_shape, sigma=input_shape[-1] / 4)
    return g_window


def _init_sliding_window(data, step_size, input_shape, in_channels, is2d, track_progress, normalization):
    if is2d:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=(1, input_shape[0], input_shape[1]),
                                         in_channels=in_channels, track_progress=track_progress,
                                         normalization=normalization)
    else:
        sw = sliding_window_multichannel(data, step_size=step_size, window_size=input_shape,
                                         track_progress=track_progress, normalization=normalization)
    return sw


def _orient(data, orientation=0):
    """
    This function essentially places the desired orientation axis to that of the original Z-axis
    For example:
          (C, Z, Y, X) -> (C, Y, Z, X) for orientation=1
          (C, Z, Y, X) -> (C, X, Y, Z) for orientation=2
    Note that applying this function twice corresponds to the identity transform

    :param data: assumed to be of shape (C, Z, Y, X)
    :param orientation: 0, 1 or 2 (respectively for Z, Y or X axis)
    :return: reoriented dataset
    """
    if orientation == 1:
        return np.transpose(data, axes=(0, 2, 1, 3))
    elif orientation == 2:
        return np.transpose(data, axes=(0, 3, 2, 1))
    else:
        return data


def _pad(data, input_shape, in_channels):
    # pad data if input shape is larger than data
    in_shape = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
    pad_width = [[0, 0], None, None, None]
    for d in range(3):
        padding = np.maximum(0, in_shape[d] - data.shape[d + 1])
        before = padding // 2
        after = padding - before
        pad_width[d + 1] = [before, after]

    # pad z-slices if necessary (required if the network uses more than 1 input channel)
    if in_channels > 1:
        c = (in_channels // 2)
        pad_width[1][0] = pad_width[1][0] + c
        pad_width[1][1] = pad_width[1][1] + c

    return np.pad(data, pad_width=pad_width, mode='symmetric'), pad_width


def _crop(data, seg_cum, counts_cum, pad_width):
    return data[:, pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]], \
           seg_cum[:, pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]], \
           counts_cum[pad_width[1][0]:data.shape[1] - pad_width[1][1], pad_width[2][0]:data.shape[2] - pad_width[2][1],
           pad_width[3][0]:data.shape[3] - pad_width[3][1]]


def _forward_prop(net, x):
    outputs = net(x)
    # if the outputs are a tuple, take the last
    if type(outputs) is tuple:
        outputs = outputs[-1]
    return F.softmax(outputs, dim=1).detach().cpu().numpy()


def _cumulate_segmentation(seg_cum, counts_cum, outputs, g_window, positions, batch_size, input_shape, in_channels,
                           is2d):
    c = in_channels // 2
    for b in range(batch_size):
        (z_b, y_b, x_b) = positions[b, :]
        # take into account the gaussian filtering
        if is2d:
            z_b += c  # correct channel shift
            seg_cum[:, z_b, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += \
                np.multiply(g_window, outputs[b, ...])
            counts_cum[z_b:z_b + 1, y_b:y_b + input_shape[0], x_b:x_b + input_shape[1]] += g_window
        else:
            seg_cum[:, z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += \
                np.multiply(g_window, outputs[b, ...])
            counts_cum[z_b:z_b + input_shape[0], y_b:y_b + input_shape[1], x_b:x_b + input_shape[2]] += g_window


def _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, in_channels,
                   is2d):
    # convert to tensors and switch to correct device
    inputs = tensor_to_device(torch.FloatTensor(batch), device)
    # forward prop
    outputs = _forward_prop(net, inputs)
    # cumulate segmentation volume
    _cumulate_segmentation(seg_cum, counts_cum, outputs, g_window, positions, batch_size, input_shape, in_channels,
                           is2d)


def segment_multichannel_2d(data, net, input_shape, batch_size=1, step_size=None, train=False, track_progress=False,
                            device=0, normalization='unit'):
    """
    Segment a multichannel 2D image using a specific network

    :param data: 3D array (C, Y, X) representing the multichannel 2D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (2-tuple)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    # make sure we compute everything on the correct device
    module_to_device(net, device)

    # set the network in the correct mode
    if train:
        net.train()
    else:
        net.eval()

    # pad data if necessary
    data, pad_width = _pad(data[:, np.newaxis, ...], input_shape, 1)
    data = data[:, 0, ...]

    # get the amount of channels
    channels = data.shape[0]

    # initialize the step size
    step_size = _init_step_size(step_size, input_shape, True)

    # gaussian window for smooth block merging
    g_window = _init_gaussian_window(input_shape, True)

    # allocate space
    seg_cum = np.zeros((net.out_channels, 1, *data.shape[1:]))
    counts_cum = np.zeros((1, *data.shape[1:]))

    # define sliding window
    sw = _init_sliding_window(data[np.newaxis, ...], [channels, *step_size[1:]], input_shape, channels, True,
                              track_progress, normalization)

    # start prediction
    batch_counter = 0
    batch = np.zeros((batch_size, channels, *input_shape))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:
            # process a single batch
            _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, 1,
                           True)

            # reset batch counter
            batch_counter = 0

    # don't forget to process the last batch
    _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, 1, True)

    # crop out the symmetric extension and compute segmentation
    data, seg_cum, counts_cum = _crop(data[:, np.newaxis, ...], seg_cum, counts_cum, pad_width)
    for c in range(net.out_channels):
        seg_cum[c, ...] = np.divide(seg_cum[c, ...], counts_cum)
    seg_cum = seg_cum[:, 0, ...]

    return seg_cum


def segment_multichannel_3d(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False,
                            track_progress=False, device=0, orientation=0, normalization='unit'):
    """
    Segment a multichannel 3D image using a specific network

    :param data: 4D array (C, Z, Y, X) representing the multichannel 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientation: orientation to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    # make sure we compute everything on the correct device
    module_to_device(net, device)

    # set the network in the correct mode
    if train:
        net.train()
    else:
        net.eval()

    # orient data if necessary
    data = _orient(data, orientation)

    # pad data if necessary
    data, pad_width = _pad(data, input_shape, in_channels)

    # 2D or 3D
    is2d = len(input_shape) == 2

    # get the amount of channels
    channels = data.shape[0]
    if is2d:
        channels = in_channels

    # initialize the step size
    step_size = _init_step_size(step_size, input_shape, is2d)

    # gaussian window for smooth block merging
    g_window = _init_gaussian_window(input_shape, is2d)

    # allocate space
    seg_cum = np.zeros((net.out_channels, *data.shape[1:]))
    counts_cum = np.zeros(data.shape[1:])

    # define sliding window
    sw = _init_sliding_window(data, step_size, input_shape, in_channels, is2d, track_progress, normalization)

    # start prediction
    batch_counter = 0
    batch = np.zeros((batch_size, channels, *input_shape))
    positions = np.zeros((batch_size, 3), dtype=int)
    for (z, y, x, inputs) in sw:

        # fill batch
        batch[batch_counter, ...] = inputs
        positions[batch_counter, :] = [z, y, x]

        # increment batch counter
        batch_counter += 1

        # perform segmentation when a full batch is filled
        if batch_counter == batch_size:
            # process a single batch
            _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape,
                           in_channels, is2d)

            # reset batch counter
            batch_counter = 0

    # don't forget to process the last batch
    _process_batch(net, batch, device, seg_cum, counts_cum, g_window, positions, batch_size, input_shape, in_channels,
                   is2d)

    # crop out the symmetric extension and compute segmentation
    data, seg_cum, counts_cum = _crop(data, seg_cum, counts_cum, pad_width)
    for c in range(net.out_channels):
        seg_cum[c, ...] = np.divide(seg_cum[c, ...], counts_cum)

    # reorient data to its original orientation
    data = _orient(data, orientation)
    seg_cum = _orient(seg_cum, orientation)

    return seg_cum


def segment_multichannel(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False,
                         track_progress=False, device=0, orientation=0, normalization='unit'):
    """
    Segment a multichannel 2D or 3D image using a specific network

    :param data: 4D array (C, [Z, ]Y, X) representing the multichannel image
    :param net: image-to-image segmentation network
    :param input_shape: size of the inputs (either 2 or 3-tuple)
    :param in_channels: amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientation: orientation to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    if data.ndim == 4:
        return segment_multichannel_3d(data, net, input_shape, in_channels=in_channels, batch_size=batch_size,
                                       step_size=step_size, train=train, track_progress=track_progress, device=device,
                                       orientation=orientation, normalization=normalization)
    else:
        return segment_multichannel_2d(data, net, input_shape, batch_size=batch_size, step_size=step_size, train=train,
                                       track_progress=track_progress, device=device, normalization=normalization)


def _segment_z_block(z_block, net, sub_block_size, overlap_size, input_shape, in_channels=1, batch_size=1,
                     step_size=None, train=False, track_progress=False, device=0, orientations=(0,),
                     normalization='unit', bar=None):
    z, y, x = z_block.shape
    yb, xb = sub_block_size[1:]
    yo, xo = overlap_size[1:]

    seg_cum = np.zeros((net.out_channels, *z_block.shape))
    wgt_cum = np.zeros(z_block.shape)
    sigma_wgt_window = np.min(sub_block_size[1:]) / 4
    weight_wnd_ = gaussian_window((sub_block_size[0], sub_block_size[2], sub_block_size[1]), sigma=sigma_wgt_window)
    for j in range(0, y, yb):
        yb_start = np.maximum(0, j - yo)
        yb_stop = np.minimum(j + yb + yo, y)
        for k in range(0, x, xb):
            xb_start = np.maximum(0, k - xo)
            xb_stop = np.minimum(k + xb + xo, x)

            block = z_block[:, yb_start:yb_stop, xb_start:xb_stop]
            segmented_block = segment(block, net, input_shape, in_channels=in_channels, batch_size=batch_size,
                                      step_size=step_size, train=train, device=device, orientations=orientations, normalization=normalization)
            if block.shape == sub_block_size:
                weight_wnd = weight_wnd_
            else:
                weight_wnd = gaussian_window((block.shape[0], block.shape[2], block.shape[1]), sigma=sigma_wgt_window)

            for c in range(net.out_channels):
                seg_cum[c, :, yb_start:yb_stop, xb_start:xb_stop] \
                    = seg_cum[c, :, yb_start:yb_stop, xb_start:xb_stop] + weight_wnd * segmented_block[c]
            wgt_cum[:, yb_start:yb_stop, xb_start:xb_stop] = wgt_cum[:, yb_start:yb_stop, xb_start:xb_stop] + weight_wnd

            if track_progress:
                bar.next()

    for c in range(net.out_channels):
        seg_cum[c] = seg_cum[c] / wgt_cum

    return seg_cum


def _cumulate_validation_metrics(z_block_pred, z_block_labels, js_cum, ams_cum, classes_of_interest):
    all_labeled = np.sum(z_block_labels == 255) == 0
    w = None if all_labeled else z_block_labels != 255

    js = np.asarray(
        [_iou_part_metrics(z_block_labels == c, z_block_pred[i], w=w) for i, c in enumerate(classes_of_interest)])
    ams = np.asarray(
        [_conf_matrix_metrics(z_block_labels == c, z_block_pred[i], w=w) for i, c in enumerate(classes_of_interest)])

    js_cum += js
    ams_cum += ams

    return js_cum, ams_cum


def _compute_validation_metrics(js_cum, ams_cum, classes_of_interest):
    js = js_cum[:, 0] / (js_cum[:, 1] + js_cum[:, 2] - js_cum[:, 0])

    total = np.sum(ams_cum, axis=1)
    accuracy = (ams_cum[:, 0] + ams_cum[:, 1] + 1) / (total + 1)
    recall = (ams_cum[:, 0] + 1) / (ams_cum[:, 0] + ams_cum[:, 3] + 1)
    specificity = (ams_cum[:, 1] + 1) / (ams_cum[:, 1] + ams_cum[:, 3] + 1)
    balanced_accuracy = (recall + specificity) / 2
    precision = (ams_cum[:, 0] + 1) / (ams_cum[:, 0] + ams_cum[:, 3] + 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    ams = np.asarray([(accuracy[i], balanced_accuracy[i], precision[i], recall[i], f1[i]) for i, c in
                      enumerate(classes_of_interest)])

    return js, ams


def _write_segmentation(segmentation, write_dir, classes_of_interest=(0, 1), index_inc=0):
    mkdir(write_dir)
    segmentation_volume = np.zeros(segmentation.shape[1:])
    for i, c in enumerate(classes_of_interest):
        segmentation_volume[segmentation[i] > 0.5] = c
    write_volume(segmentation_volume, write_dir, index_inc=index_inc, type='pngseq')


def _validate_ram(segmentation, labels, classes_of_interest=(0, 1), hausdorff=False, report=True):
    """
    Validate a segmentation based on available labels

    :param segmentation: 4D array (C, Z, Y, X) representing the a class probability distribution over the 3D image
    :param labels: 3D array (Z, Y, X) representing the 3D labels
    :param classes_of_interest: index of the label of interest
    :param hausdorff: compute hausdorff or not
    :param report: print the validation report or not (only required if labels is not None)
    :return: validation results, i.e. accuracy, precision, recall, f-score, jaccard and dice score
    """

    # compute metrics
    all_labeled = np.sum(labels == 255) == 0
    w = None if all_labeled else labels != 255
    comp_hausdorff = all_labeled and hausdorff
    js = np.asarray([iou(labels == c, segmentation[i], w=w) for i, c in enumerate(classes_of_interest)])
    ams = np.asarray([accuracy_metrics(labels == c, segmentation[i], w=w) for i, c in enumerate(classes_of_interest)])
    hs = np.zeros_like(js)
    for i, c in enumerate(classes_of_interest):
        if comp_hausdorff:
            hs[i] = hausdorff_distance(labels == c, segmentation[i])[0]
        else:
            hs[i] = -1

    # report results if necessary
    if report:
        _report_validation(js, ams, classes_of_interest, hs=hs)

    return js, ams


def _report_validation(js, ams, classes_of_interest, hs=None):
    for i, c in enumerate(classes_of_interest):
        print_frm('Validation performance for class %d: ' % c)
        print_frm('    - Accuracy: %f' % ams[i, 0])
        print_frm('    - Balanced accuracy: %f' % ams[i, 1])
        print_frm('    - Precision: %f' % ams[i, 2])
        print_frm('    - Recall: %f' % ams[i, 3])
        print_frm('    - F1 score: %f' % ams[i, 4])
        print_frm('    - IoU: %f' % js[i])
        if hs is not None:
            print_frm('    - Hausdorff distance: %f' % hs[i])

    print_frm('Validation performance mean: ')
    print_frm('    - Accuracy: %f' % np.mean(ams[:, 0]))
    print_frm('    - Balanced accuracy: %f' % np.mean(ams[:, 1]))
    print_frm('    - Precision: %f' % np.mean(ams[:, 2]))
    print_frm('    - Recall: %f' % np.mean(ams[:, 3]))
    print_frm('    - F1 score: %f' % np.mean(ams[:, 4]))
    print_frm('    - mIoU: %f' % np.mean(js))
    if hs is not None:
        print_frm('    - Hausdorff distance: %f' % np.mean(hs))


def _save_validation(js, ams, val_file):
    np.save(val_file, np.concatenate((js[:, np.newaxis], ams), axis=1))


def segment_read(data, net, input_shape, write_dir, start=0, stop=-1, block_size=(50, 1024, 1024),
                 overlap_size=(16, 128, 128), in_channels=1, batch_size=1, step_size=None, train=False,
                 track_progress=False, device=0, orientations=(0,), normalization='unit', type='pngseq', labels=None,
                 val_file=None, classes_of_interest=(0, 1), report=True):
    """
    Segment a 3D image using a specific network based on a directory of data.
    This function is designed for large datasets where RAM becomes a bottleneck. The segmentation is performed with
    overlapping blocks and optional writing/validation is also performed on a block basis. Note that the hausdorff
    distance metric is not supported du to block-based processing

    :param data: directory that contains a 3D array (Z, Y, X) representing the 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of
    :param start: start position to segment (0 for beginning of the data)
    :param stop: stop position to segment (-1 for end of the data)
    :param block_size: block size for processing
    :param overlap_size: overlap size for avoiding blocking artifacts
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param write_dir: destination for the segmentation, set this equal to None, in which case only validation will be performed
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientations: list of orientations to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :param type: type of the data (pngseq or tifseq)
    :param labels: directory that contains a 3D array (Z, Y, X) representing the 3D labels
    :param val_file: destination of the validation file (only required if labels is not None)
    :param classes_of_interest: index of the label of interest (required if labels is not None or write_dir is not None)
    :param report: print the validation report or not (only required if labels is not None)
    """

    # get the dimensions of the data
    files = os.listdir(data)
    z = len(files)
    y, x = read_volume(data, type=type, start=0, stop=1).shape[1:]

    # get the start and stop positions
    if stop < 0:
        stop = z
    z = stop - start

    # the block size
    zb, yb, xb = block_size
    zo, yo, xo = overlap_size

    # initialize the validation metrics
    js_cum = np.zeros((len(classes_of_interest), 3))
    ams_cum = np.zeros((len(classes_of_interest), 4))

    # initialize progress bar
    if track_progress:
        bar = Bar('Progress', max=len(range(0, z, zb)) * len(range(0, y, yb)) * len(range(0, x, xb)))
    else:
        bar = None

    # first check whether there are at least zb slices
    # if not, we can straightforward segment the z-block
    if z <= zb:
        z_block = read_volume(data, type=type, start=start, stop=stop)
        z_block_segmented = _segment_z_block(z_block, net, (zb, yb, xb), (zo, yo, xo), input_shape,
                                             in_channels=in_channels, batch_size=batch_size, step_size=step_size,
                                             train=train, track_progress=track_progress, device=device,
                                             orientations=orientations, normalization=normalization, bar=bar)
        if write_dir is not None:
            _write_segmentation(z_block_segmented, write_dir, classes_of_interest=classes_of_interest, index_inc=start)
        if labels is not None:
            z_block_labels = read_volume(labels, type=type, start=start, stop=stop)
            js_cum, ams_cum = _cumulate_validation_metrics(z_block_segmented, z_block_labels, js_cum, ams_cum,
                                                           classes_of_interest)

    else:

        # segment the first z-block
        z_block = read_volume(data, type=type, start=start, stop=start + zb)
        z_block_segmented = _segment_z_block(z_block, net, (zb, yb, xb), (zo, yo, xo), input_shape,
                                             in_channels=in_channels,
                                             batch_size=batch_size, step_size=step_size, train=train,
                                             track_progress=track_progress, device=device, orientations=orientations,
                                             normalization=normalization, bar=bar)

        # process the remaining z-blocks
        j = 0
        for i in range(zb, z, zb):
            z_start = i - zo
            z_stop = np.minimum(i + zb, z)
            z_block_prev = z_block_segmented
            z_block = read_volume(data, type=type, start=start + z_start, stop=start + z_stop)
            # overlapping region is in slices zb-zo through zb in z_block_prev and 0 through zo in z_block

            # segment z_block
            z_block_segmented = _segment_z_block(z_block, net, (zb, yb, xb), (zo, yo, xo), input_shape,
                                                 in_channels=in_channels, batch_size=batch_size, step_size=step_size,
                                                 train=train, track_progress=track_progress, device=device,
                                                 orientations=orientations, normalization=normalization, bar=bar)

            # merge overlapping regions in two z_blocks
            w = np.arange(1, zo + 1, 1) / zo
            w = np.ones((y, x))[np.newaxis, ...] * w[:, np.newaxis, np.newaxis]
            w_ = 1 - w
            for c in range(net.out_channels):
                z_block_prev[c, -zo:] = w * z_block_prev[c, -zo:] + w_ * z_block_segmented[c, :zo]

            # z_block_prev is completely segmented, ready to write out and validate
            if write_dir is not None:
                if i == zb:
                    _write_segmentation(z_block_prev, write_dir, classes_of_interest=classes_of_interest,
                                        index_inc=start)
                else:
                    _write_segmentation(z_block_prev[:, zo:], write_dir, classes_of_interest=classes_of_interest,
                                        index_inc=start + j * zb)
            if labels is not None:
                if i == zb:
                    z_block_labels = read_volume(labels, type=type, start=start, stop=start + zb)
                    js_cum, ams_cum = _cumulate_validation_metrics(z_block_prev, z_block_labels, js_cum, ams_cum,
                                                                   classes_of_interest)
                else:
                    z_block_labels = read_volume(labels, type=type, start=start + j * zb, stop=start + (j + 1) * zb)
                    js_cum, ams_cum = _cumulate_validation_metrics(z_block_prev[:, zo:], z_block_labels, js_cum,
                                                                   ams_cum, classes_of_interest)

            j += 1

        # write out and validate last block
        if write_dir is not None:
            _write_segmentation(z_block_segmented[:, zo:], write_dir, classes_of_interest=classes_of_interest,
                                index_inc=start + j * zb)
        if labels is not None:
            z_block_labels = read_volume(labels, type=type, start=start + j * zb, stop=start + z)
            js_cum, ams_cum = _cumulate_validation_metrics(z_block_segmented[:, zo:], z_block_labels, js_cum, ams_cum,
                                                           classes_of_interest)

    # compute validation metrics
    js, ams = _compute_validation_metrics(js_cum, ams_cum, classes_of_interest)

    # report validation metrics if necessary
    if track_progress:
        print()
    if report:
        _report_validation(js, ams, classes_of_interest)

    # save the results if necessary
    if val_file is not None:
        _save_validation(js, ams, val_file)

    return js, ams


def segment_ram(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False, track_progress=False,
                device=0, orientations=(0,), normalization='unit', write_dir=None, labels=None, val_file=None,
                classes_of_interest=(0, 1), hausdorff=False, report=True):
    """
    Segment a 3D image using a specific network

    :param data: 3D array (Z, Y, X) representing the 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientations: list of orientations to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :param write_dir: destination for the segmentation, set this equal to None, in which case only validation will be performed
    :param labels: directory that contains a 3D array (Z, Y, X) representing the 3D labels
    :param val_file: destination of the validation file (only required if labels is not None)
    :param classes_of_interest: index of the label of interest (required if labels is not None or write_dir is not None)
    :param hausdorff: compute hausdorff or not (only required if labels is not None)
    :param report: print the validation report or not (only required if labels is not None)
    :return: the segmented image
    """

    # compute segmentation for each orientation and average results
    segmentation = segment(data, net, input_shape, in_channels=in_channels, batch_size=batch_size, step_size=step_size,
                           train=train, track_progress=track_progress, device=device, orientations=orientations,
                           normalization=normalization)

    # write out the segmentation if necessary
    if write_dir is not None:
        _write_segmentation(segmentation, write_dir, classes_of_interest=classes_of_interest)

    # validate the segmentation if necessary
    if labels is not None:
        js, ams = _validate_ram(segmentation, labels, classes_of_interest=classes_of_interest, hausdorff=hausdorff,
                                report=report)
        # save the results if necessary
        if val_file is not None:
            _save_validation(js, ams, val_file)
    else:
        js, ams = None, None

    return segmentation, js, ams


def segment(data, net, input_shape, in_channels=1, batch_size=1, step_size=None, train=False, track_progress=False,
            device=0, orientations=(0,), normalization='unit'):
    """
    Segment a 3D image using a specific network.
    This function assumes there is plenty of RAM memory available to store the volume, its segmentation and all
    remaining temporary variables. For large volumes this can become quite memory intensive. In that case, please
    consider the segment_read function.

    :param data: 3D array (Z, Y, X) representing the 3D image
    :param net: image-to-image segmentation network
    :param input_shape: size of
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param batch_size: batch size for processing
    :param step_size: step size of the sliding window
    :param train: evaluate the network in training mode
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientations: list of orientations to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :return: the segmented image
    """

    # compute segmentation for each orientation and average results
    segmentation = np.zeros((net.out_channels, *data.shape))
    for orientation in orientations:
        segmentation += segment_multichannel(data[np.newaxis, ...], net, input_shape, in_channels=in_channels,
                                             batch_size=batch_size, step_size=step_size, train=train,
                                             track_progress=track_progress, device=device, orientation=orientation,
                                             normalization=normalization)
    segmentation = segmentation / len(orientations)

    return segmentation


def validate(net, data, labels, input_size, in_channels=1, classes_of_interest=(0, 1), batch_size=1, write_dir=None,
             val_file=None, track_progress=False, device=0, orientations=(0,), normalization='unit', hausdorff=False,
             report=True):
    """
    Validate a network on a dataset and its labels

    :param net: image-to-image segmentation network
    :param data: 3D array (Z, Y, X) representing the 3D image, or a directory that specifies to it (if RAM is limited)
    :param labels: 3D array (Z, Y, X) representing the 3D labels, or a directory that specifies to it (if RAM is limited)
    :param input_size: size of the inputs (either 2 or 3-tuple) for processing
    :param in_channels: Amount of subsequent slices that serve as input for the network (should be odd)
    :param classes_of_interest: index of the label of interest
    :param batch_size: batch size for processing
    :param write_dir: optionally, specify a directory to write the output
    :param val_file: optionally, specify a file to write the validation results
    :param track_progress: optionally, for tracking progress with progress bar
    :param device: GPU device where the computations should occur
    :param orientations: list of orientations to perform segmentation: 0-Z, 1-Y, 2-X (only for 2D based segmentation)
    :param normalization: type of data normalization (unit, z or minmax)
    :param hausdorff: compute hausdorff or not
    :param report: get a report of the validation results or not
    :return: validation results, i.e. accuracy, precision, recall, f-score, jaccard and dice score
    """

    if data.__class__ == str:
        js, ams = segment_read(data, net, input_size, write_dir=write_dir, in_channels=in_channels,
                               batch_size=batch_size, track_progress=track_progress, device=device,
                               orientations=orientations, normalization=normalization, labels=labels, val_file=val_file,
                               classes_of_interest=classes_of_interest, report=report)
    else:
        segmentation, js, ams = segment_ram(data, net, input_size, write_dir=write_dir, in_channels=in_channels,
                                            batch_size=batch_size, track_progress=track_progress, device=device,
                                            orientations=orientations, normalization=normalization, labels=labels,
                                            val_file=val_file, classes_of_interest=classes_of_interest,
                                            hausdorff=hausdorff, report=report)

    return js, ams
