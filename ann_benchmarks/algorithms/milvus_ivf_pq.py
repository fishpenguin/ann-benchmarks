from __future__ import absolute_import
import time
import milvus
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.milvus_ivf_flat import MilvusIVFFLAT


class MilvusIVFPQ(MilvusIVFFLAT):
    def __init__(self, metric, dataset, index_type, nlist, m):
        self._index_param = {'nlist': nlist, 'm': m}
        self._search_param = {'nprobe': None}
        self._metric = {'angular': milvus.MetricType.IP, 'euclidean': milvus.MetricType.L2}[metric]
        self._milvus = milvus.Milvus(host='172.16.0.53', port='19530', try_connect=False, pre_ping=False)
        # import uuid
        # self._table_name = 'test_' + str(uuid.uuid1()).replace('-', '_')
        self._table_name = dataset.replace('-', '_')
        self._index_file_size = 2048
        postfix = '_' + str(metric) + '_' + str(index_type) + '_' + str(nlist) + '_' + str(m) + '_' + str(self._index_file_size)
        self._table_name += postfix
        self._table_name.replace('-', '_')
        self._index_type = index_type

        # batch fit
        self._already_nums = 0

        # batch search
        self._res = None

    def __str__(self):
        return 'MilvusIVFPQ(index={}, index_param={}, search_param={})'.format(self._index_type, self._index_param, self._search_param)
