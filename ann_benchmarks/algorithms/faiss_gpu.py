from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
# sys.path.append("faiss")  # noqa
import numpy
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.algorithms.faiss import (FaissIVF,
                                             FaissIVFPQ,
                                             FaissIVFSQ)

# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


class FaissGPU(BaseANN):
    def __init__(self, n_bits, n_probes):
        self.name = 'FaissGPU(n_bits={}, n_probes={})'.format(
            n_bits, n_probes)
        self._n_bits = n_bits
        self._n_probes = n_probes
        self._res = faiss.StandardGpuResources()
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        self._index = faiss.GpuIndexIVFFlat(self._res, len(X[0]), self._n_bits,
                                            faiss.METRIC_L2)
        # self._index = faiss.index_factory(len(X[0]),
        #                                   "IVF%d,Flat" % self._n_bits)
        # co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
        # self._index = faiss.index_cpu_to_gpu(self._res, 0,
        #                                      self._index, co)
        self._index.train(X)
        self._index.add(X)
        self._index.setNumProbes(self._n_probes)

    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        v = v.astype(numpy.float32).reshape(1, -1)
        distances, labels = self._index.search(v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            if l != -1:
                r.append((l, d))
        return r

    def batch_query(self, X, n):
        self.res = self._index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

class FaissGPUIVF(FaissIVF):
    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        
        dimension = X.shape[1]
        index = faiss.GpuIndexIVFFlat(
            faiss.StandardGpuResources(),
            dimension,
            self._n_list,
            faiss.METRIC_L2,
        )
        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissGPUIVF(n_list=%d, n_probe=%d)' % (self._n_list,
                                                       self._n_probe)

class FaissGPUIVFPQ(FaissIVFPQ):
    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        dimension = X.shape[1]
        index = faiss.GpuIndexIVFPQ(
            faiss.StandardGpuResources(),
            dimension,
            self._n_list,
            self._m,
            self._n_bits,
            faiss.METRIC_L2,
        )
        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissGPUIVFPQ(n_list=%d, n_probe=%d, m=%d)' % (
            self._n_list,
            self._n_probe,
            self._m
        )

class FaissGPUOIVFPQ(FaissGPUIVFPQ):
    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        dimension = X.shape[1]
        index = faiss.GpuIndexIVFPQ(
            faiss.StandardGpuResources(),
            dimension,
            self._n_list,
            self._m,
            self._n_bits,
            faiss.METRIC_L2,
        )
        opq_matrix = faiss.OPQMatrix(dimension, self._m)
        # opq_matrix.niter = 10 # TODO find how this parameter works
        index = faiss.IndexPreTransform(opq_matrix, index)
        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissGPUOIVFPQ(n_list=%d, n_probe=%d, m=%d)' % (
            self._n_list,
            self._n_probe,
            self._m
        )

class FaissGPUIVFSQ(FaissIVFSQ):
    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        dimension = X.shape[1]
        qtype = getattr(faiss.ScalarQuantizer, self._qname)
        index = faiss.GpuIndexIVFScalarQuantizer(
            faiss.StandardGpuResources(),
            dimension,
            self._n_centroids,
            qtype,
            faiss.METRIC_L2
        )
        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissGPUIVFSQ(n_probe=%d, qname=%s, n_centroids=%d)' % (
            self._n_probe,
            self._qname,
            self._n_centroids
        )
