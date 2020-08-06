#!/usr/bin/env python

import os
import struct
import numpy as np

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_to_ndarray(bvecs_fn):
    with open(bvecs_fn, 'rb') as f:
        fsize = os.path.getsize(bvecs_fn)
        dimension, = struct.unpack('i', f.read(4))
        vector_nums = fsize // (4 + dimension)

        v = np.zeros((vector_nums, dimension))
        for i in range(vector_nums):
            f.read(4)
            v[i] = struct.unpack('B' * dimension, f.read(dimension))
        
        return v
