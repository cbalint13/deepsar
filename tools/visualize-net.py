#!/usr/bin/python

import os
import sys
import cv2
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

def vis_square(data, padsize=1, padval=0):

    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    print data.shape

    data[:,:,0] = (data[:,:,0] + data[:,:,3])/2

    plt.imshow(data[:,:,0:3], origin='upper')
    plt.draw()
    plt.waitforbuttonpress(0)

# last model
h5mod = max(["./snapshots/" + f for f in os.listdir("./snapshots/") if f.lower().endswith('.caffemodel.h5')], key=os.path.getctime)

print h5mod

h5f = h5py.File(h5mod)
filters = h5f['/data/Convolution6/0'][...]

print filters.shape

vis_square( filters.transpose(0, 2, 3, 1) )
