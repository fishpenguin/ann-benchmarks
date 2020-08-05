#!/usr/bin/env python

import os
import struct
import numpy as np

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
