from __future__ import absolute_import
import time
import milvus
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.milvus_ivf_flat import MilvusIVFFLAT


class MilvusIVFPQ(MilvusIVFFLAT):
    def __str__(self):
        return 'MilvusIVFPQ(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)
