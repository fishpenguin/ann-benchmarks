#!/usr/bin/env python

import os
import struct
import numpy as np
import time
import h5py
import sys

sift_prefix = '/cifs/data/milvus_paper/sift'
train_size_map = {
    "1M": 1000000,
    "2M": 2000000,
    "5M": 5000000,
    "10M": 10000000,
    "20M": 20000000,
    "50M": 50000000,
    "100M": 100000000,
    "200M": 200000000,
    "500M": 500000000,
    "1000M": 1000000000,
}

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_to_ndarray(bvecs_fn, size=sys.maxsize):
    with open(bvecs_fn, 'rb') as f:
        fsize = os.path.getsize(bvecs_fn)
        dimension, = struct.unpack('i', f.read(4))
        print('dimension: ', dimension)
        vector_nums = fsize // (4 + dimension)
        vector_nums = min(vector_nums, size)
        print('vector_nums: ', vector_nums)

        v = np.zeros((vector_nums, dimension))
        f.seek(0)
        begin = time.time()
        for i in range(vector_nums):
            if i % 10000 == 0:
                print("handle %dth vector, time cost: %d" % (i, time.time() - begin))
            f.read(4)
            v[i] = struct.unpack('B' * dimension, f.read(dimension))
        
        return v

def handle_sift_1b(out_fn, size='1000M'):
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = 'euclidean'
    f.attrs['point_type'] = 'float'

    idx_file = '/data1/workspace/milvus_data/sift_data/gnd/idx_{}.ivecs'.format(size)
    dis_file = '/data1/workspace/milvus_data/sift_data/gnd/dis_{}.fvecs'.format(size)

    neighbors = ivecs_read(idx_file)
    distances = fvecs_read(dis_file)
    train = bvecs_to_ndarray('/data1/workspace/milvus_data/sift_data/bigann_base.bvecs', train_size_map[size])
    test = bvecs_to_ndarray('/data1/workspace/milvus_data/sift_data/bigann_query.bvecs')
    dimension = len(test[0])
    assert len(neighbors) == len(distances) == len(test)
    count = len(neighbors[0])

    f.create_dataset(
        'test',
        (len(test), dimension),
        dtype=test.dtype,
    )[:] = test
    f.create_dataset(
        'train',
        (len(train), dimension),
        dtype=train.dtype,
    )[:] = train
    f.create_dataset('neighbors', (len(test), count), dtype='i')[:] = neighbors
    f.create_dataset('distances', (len(test), count), dtype='f')[:] = distances

    f.close()

def handle_sift_1b_single(out_fn, size='1000M'):
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = 'euclidean'
    f.attrs['point_type'] = 'float'

    idx_file = sift_prefix + '/gnd/idx_{}.ivecs'.format(size)
    dis_file = sift_prefix + '/gnd/dis_{}.fvecs'.format(size)

    neighbors = ivecs_read(idx_file)
    distances = fvecs_read(dis_file)
    test = bvecs_to_ndarray(sift_prefix + '/bigann_query.bvecs')
    dimension = len(test[0])
    assert len(neighbors) == len(distances) == len(test)
    count = len(neighbors[0])

    f.create_dataset('neighbors', (len(test), count), dtype='i')[:] = neighbors
    f.create_dataset('distances', (len(test), count), dtype='f')[:] = distances
    f.create_dataset(
        'test',
        (len(test), dimension),
        dtype=test.dtype,
    )[:] = test

    train_fn = sift_prefix + '/bigann_base.bvecs'
    with open(train_fn, 'rb') as ftrain:
        train_size = os.path.getsize(train_fn)
        train_dimension, = struct.unpack('i', ftrain.read(4))
        print("train_dimension: ", train_dimension)
        assert train_dimension == dimension
        vector_nums = train_size // (4 + train_dimension)
        vector_nums = min(vector_nums, train_size_map[size])
        print("vector_nums: ", vector_nums)
        train = f.create_dataset(
            'train',
            (vector_nums, dimension),
            dtype=test.dtype,
        )
        begin = time.time()
        ftrain.seek(0)
        for i in range(vector_nums):
            train_dimension, = struct.unpack('i', ftrain.read(4))
            assert train_dimension == dimension
            train[i] = struct.unpack('B' * train_dimension, ftrain.read(train_dimension))
            if i % 100000 == 0:
                print("handle %dth vector, time cost: %d" % (i, time.time() - begin))

    f.close()

def handle_deep_1b(out_fn):
    import sklearn.preprocessing
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = 'angular'
    f.attrs['point_type'] = 'float'
    ground_truth = ivecs_read('/cifs/data/milvus_paper/deep1b/deep1B_groundtruth.ivecs')
    test = fvecs_read('/cifs/data/milvus_paper/deep1b/deep1B_queries.fvecs')
    test = sklearn.preprocessing.normalize(test, axis=1, norm='l2')
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
            ftrain.read(4) # ignore dimension, faster
            #train_dimension, = struct.unpack('i', ftrain.read(4))
            #assert train_dimension == dimension
            train[i] = struct.unpack('f' * train_dimension, ftrain.read(train_dimension * 4))
            #lens = (train[i] ** 2).sum(-1)
            #train[i] /= np.sqrt(lens)[..., np.newaxis]
            # below is faster
            train[i] = sklearn.preprocessing.normalize([train[i]], axis=1, norm='l2')[0]
            #print('type(train[i]: ', type(train[i]))
            #print((train[i] ** 2).sum(-1))
            #assert (train[i] ** 2).sum(-1) == 1.0
            if i % 100000 == 0:
                print("handle %dth vector, time cost: %d" % (i, time.time() - begin))

    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    for i, x in enumerate(test):
        distances[i] = [np.dot(train[idx], x) for idx in ground_truth[i]]
    f.close()

def main():
    #handle_deep_1b('/cifs/data/milvus_paper/deep1b/deep-1b-angular.hdf5')
    #handle_sift_1b('./sift-1b-euclidean.hdf5')
    handle_sift_1b_single(sift_prefix + '/sift-1b-euclidean.hdf5', size='1000M')

if __name__ == "__main__":
    main()
