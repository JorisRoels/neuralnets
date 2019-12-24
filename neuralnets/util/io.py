import os

import h5py
import numpy as np
import tifffile as tiff
import cv2


def read_volume(file, type='tif3d', key=None):
    """
    Reads a volume file/directory and returns the data in it as a numpy array
    :param file: path to the data
    :param type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param key: key to the data (only necessary for hdf5 files)
    :return: numpy array containing the data
    """

    if type == 'tif2d' or type == 'tif3d':
        volume = read_tif(file)
    elif type == 'tifseq':
        volume = read_tifseq(file)
    elif type == 'hdf5':
        volume = read_hdf5(file, key=key)
    elif type == 'png':
        volume = read_png(file)
    elif type == 'pngseq':
        volume = read_pngseq(file)
    else:
        volume = None

    return volume


def read_tif(file, dtype='uint8'):
    """
    Reads tif formatted file and returns the data in it as a numpy array
    :param file: path to the tif file
    :param dtype: data type of the numpy array
    :return: numpy array containing the data
    """

    data = tiff.imread(file).astype(dtype)

    return data


def read_tifseq(dir, dtype='uint8'):
    """
    Read a sequence of 2D TIF files
    :param dir: directory that contains the files
    :param dtype: data type of the output
    """

    files = os.listdir(dir)
    files.sort()
    sz = tiff.imread(os.path.join(dir, files[0])).shape
    data = np.zeros((len(files), sz[0], sz[1]), dtype=dtype)
    for i, file in enumerate(files):
        data[i] = tiff.imread(os.path.join(dir, file))

    return data


def read_hdf5(file, dtype='uint8', key=None):
    """
    Reads an HDF5 file as a numpy array
    :param file: path to the hdf5 file
    :param dtype: data type of the numpy array
    :param key: key in the hfd5 file that provides access to the data
    :return: numpy array containing the data
    """
    f = h5py.File(file, 'r')
    data = np.array(f.get(key), dtype=dtype)
    f.close()

    return data


def read_png(file, dtype='uint8'):
    """
    Read a 2D PNG file
    :param file: file to be read
    :param dtype: data type of the output
    """

    data = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(dtype)

    return data


def read_pngseq(dir, dtype='uint8'):
    """
    Read a sequence of 2D PNG files
    :param dir: directory that contains the files
    :param dtype: data type of the output
    """

    files = os.listdir(dir)
    files.sort()
    sz = cv2.imread(os.path.join(dir, files[0]), cv2.IMREAD_GRAYSCALE).shape
    data = np.zeros((len(files), sz[0], sz[1]), dtype=dtype)
    for i, file in enumerate(files):
        data[i] = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE).astype(dtype)

    return data


def write_volume(data, file, type='tif3d', z_start=0, K=4):
    """
    Writes a numpy array to a volume file/directory
    :param data: 2D/3D numpy array
    :param file: path to the data
    :param type: type of the volume file (tif2d, tif3d, tifseq, png or pngseq)
    :param z_start: starting index for writing (optional, only for sequences)
    :param K: length of the string index (optional, only for sequences)
    """

    if type == 'tif2d' or type == 'tif3d':
        write_tif(data, file)
    elif type == 'tifseq':
        write_tifseq(data, file, z_start=z_start, K=K)
    elif type == 'png':
        write_png(data, file)
    elif type == 'pngseq':
        write_pngseq(data, file, z_start=z_start, K=K)


def write_tif(x, file, dtype='uint8'):
    """
    Write a 2D/3D data set as a TIF file
    :param x: 2D/3D data array
    :param file: directory to write the data to
    :param dtype: data type of the output
    """

    tiff.imsave(file, x.astype(dtype))


def write_png(x, file):
    """
    Write a 2D data set to a PNG file
    :param x: 3D data array
    :param file: directory to write the data to
    """

    cv2.imwrite(file, x.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])


def write_tifseq(x, dir, prefix='', z_start=0, dtype='uint8', K=4):
    """
    Write a 3D data set to a directory, slice by slice, as TIF files
    :param x: 3D data array
    :param dir: directory to write the data to
    :param prefix: prefix of the separate files
    :param z_start: starting index
    :param dtype: data type of the output
    :param K: number of digits for the index
    """

    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(z_start, z_start + x.shape[0]):
        i_str = num2str(i, K=K)
        tiff.imsave(dir + '/' + prefix + i_str + '.tif', (x[i - z_start, :, :]).astype(dtype))


def write_pngseq(x, dir, prefix='', z_start=0, K=4):
    """
    Write a 3D data set to a directory, slice by slice, as PNG files
    :param x: 3D data array
    :param dir: directory to write the data to
    :param prefix: prefix of the separate files
    :param z_start: starting index
    """

    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(z_start, z_start + x.shape[0]):
        i_str = num2str(i, K=K)
        cv2.imwrite(dir + '/' + prefix + i_str + '.png', (x[i - z_start, :, :]).astype('uint8'),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])


def num2str(n, K=4):
    n_str = str(n)
    for k in range(0, K - len(n_str)):
        n_str = '0' + n_str
    return n_str
