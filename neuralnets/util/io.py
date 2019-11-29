import os

import h5py
import numpy as np
import tifffile as tiff


def read_tif(file, dtype='uint8'):
    """
    Reads tif formatted file and returns the data in it as a numpy array
    :param file: path to the tif file
    :param dtype: data type of the numpy array
    :return: numpy array containing the data
    """

    data = tiff.imread(file).astype(dtype)

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


def write_tif(file, data):
    """
    Write tif file from a numpy array
    :param file: the file to be written
    :param data: path to the tif file
    """

    tiff.imwrite(file, data)


def imwrite3D(x, dir, prefix='', rescale=False, z_start=0):
    """
    Write a 3D data set to a directory, slice by slice, as tif files
    :param x: 3D data array
    :param dir: directory to write the data to
    :param prefix: prefix of the separate files
    :param rescale: rescale the data if necessary
    :param z_start: starting index
    """

    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(z_start, z_start + x.shape[0]):
        if rescale:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i - z_start, :, :] * 255).astype('uint8'))
        else:
            tiff.imsave(dir + '/' + prefix + str(i) + '.tif', (x[i - z_start, :, :]).astype('uint8'))
