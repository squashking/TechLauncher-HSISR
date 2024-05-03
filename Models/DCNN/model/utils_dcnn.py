# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy import ndimage
import scipy

np.seterr(divide='ignore', invalid='ignore')


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial(filter_type, *args, **kwargs):
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)


def degradation_bsrgan(img):
    img = img.transpose(1, 2, 0)
    size = random.randint(3, 5)
    k = fspecial('gaussian', size, random.uniform(0.1, 0.6))
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img.transpose(2, 0, 1).astype(np.float32)
