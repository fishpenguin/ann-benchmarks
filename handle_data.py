#!/usr/bin/env python

import os
import struct
import numpy as np
import time
import h5py
import sklearn.preprocessing

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

def handle_deep_1b(out_fn):
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = 'angular'
    f.attrs['point_type'] = 'float'
    ground_truth = ivecs_read('/cifs/data/milvus_paper/deep1b/deep1B_groundtruth.ivecs')
    test = fvecs_read('/cifs/data/milvus_paper/deep1b/deep1B_queries.fvecs')
    dimension = len(test[0])
    assert len(ground_truth) == len(test)
    count = len(ground_truth[0]) # top k
    f.create_dataset(
        'test',
        (len(test), dimension),
        dtype=test.dtype,
    )[:] = test
    f.create_dataset('neighbors', (len(test), count), dtype='i')[:] = ground_truth

    # no enough memory to directly use fvecs_read
    train_fn = '/cifs/data/milvus_paper/deep1b/base/base.fvecs'
    with open(train_fn, 'rb') as ftrain:
        train_size = os.path.getsize(train_fn)
        train_dimension, = struct.unpack('i', ftrain.read(4))
        print("train_dimension: ", train_dimension)
        assert train_dimension == dimension
        vector_nums = train_size // (4 + 4 * train_dimension)
        print("vector_nums: ", vector_nums)
        train = f.create_dataset(
            'train',
            (vector_nums, dimension),
            dtype=test.dtype,
        )
        begin = time.time()
        ftrain.seek(0)
        for i in range(vector_nums):
            ftrain.read(4)
            train[i] = struct.unpack('f' * train_dimension, ftrain.read(train_dimension * 4))
            if i % 100000 == 0:
                print("handle %dth vector, time cost: %d" % (i, time.time() - begin))

    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    for i, x in enumerate(test):
        distances[i] = [np.dot(train[idx], x) for idx in ground_truth[i]]
    f.close()

def main():
    handle_deep_1b('/cifs/data/milvus_paper/deep1b/deep-1b-angular.hdf5')

if __name__ == "__main__":
    main()
