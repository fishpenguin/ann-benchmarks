#!/usr/bin/env python

from __future__ import absolute_import
import h5py
import numpy
from ann_benchmarks.algorithms.vearch import Vearch

def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor[0] in std:
            hit_nums += 1
    return hit_nums / len(answer)

def main():
    dataset = 'sift-128-euclidean'
    dataset = 'sift-10000-10'
    client = Vearch(512, 200)
    f = h5py.File('data/' + dataset + '.hdf5', 'r')
    qs = numpy.array(f['test'])
    topk = 100
    client.batch_query(qs, topk)
    ids = client.get_batch_results()
    print(ids)
    stds = numpy.array(f['neighbors'])
    client.done()

if __name__ == "__main__":
    main()
