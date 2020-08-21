#!/usr/bin/env python

from __future__ import absolute_import
import psycopg2
from psycopg2 import extras
import asyncio
import asyncpg
import h5py
import numpy
from ann_benchmarks.algorithms.pase import PaseIVFFLATAsync

def compute_recall(std, answer):
    hit_nums = 0.0
    for neighbor in answer:
        if neighbor[0] in std:
            hit_nums += 1
    return hit_nums / len(answer)

def main():
    dataset = 'sift-128-euclidean'
    client = PaseIVFFLATAsync(
        dataset,
        1000,
        host='pgm-bp174758o17b9a8ylo.pg.rds.aliyuncs.com',
    )
    f = h5py.File('data/' + dataset + '.hdf5', 'r')
    qs = numpy.array(f['test'])
    topk = 100
    client.set_query_arguments(1000)
    client.batch_query(qs, topk)
    # client.batch_query_sync(qs, topk)
    ids = client.get_batch_results()
    stds = numpy.array(f['neighbors'])
    # print(ids)
    recall_sum = 0.0
    for answer, std in zip(ids, stds):
        recall = compute_recall(std, answer)
        print('recall: ', recall)
        recall_sum += recall
    print('average recall: ', recall_sum / len(ids))

if __name__ == "__main__":
    main()
