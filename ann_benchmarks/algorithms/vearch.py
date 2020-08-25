from __future__ import absolute_import
import json
import time
import milvus
import numpy
import requests
from ann_benchmarks.algorithms.base import BaseANN

class Vearch(BaseANN):
    def __init__(self, ncentroids, nprobe, nsubvector=64, partition_num=1, replica_num=1, metric_type='L2'):
        self._db_name = 'annbench'
        self._table_name = 'annbench'
        self._field = 'field1'
        self._master_host = 'localhost'
        self._master_port = '443'
        self._router_host = 'localhost'
        self._router_port = '88'
        self._master_prefix = 'http://' + self._master_host + ':' + self._master_port
        self._router_prefix = 'http://' + self._router_host + ':' + self._router_port
        self._ncentroids = ncentroids
        self._nprobe = nprobe
        self._nsubvector = 64
        self._partition_num = partition_num
        self._replica_num = replica_num
        self._metric_type = metric_type

    def _create_db(self):
        url = self._master_prefix + '/db/_create'
        response = requests.put(url, json={'name': self._db_name})
        print("post: ", url, ", status: ", response.status_code)

    def _drop_db(self):
        url = self._master_prefix + '/db/' + self._db_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)

    def _create_table(self, dimension, ncentroids, nprobe, nsubvector=64, partition_num=1, replica_num=1, metric_type='L2'):
        payload = {
            "name": self._table_name,
            "partition_num": partition_num, # 数据分片数量，设为 PS 的数量比较合适
            "replica_num": replica_num, # 无副本，测量性能的话没必要高可用？
            "engine": {
                "name": "gamma",
                "ncentroids": ncentroids,
                "nprobe": nprobe,
                "metric_type": metric_type,
                "nsubvector": nsubvector
            },
            "properties": {
                self._field: {
                    "type": "vector",
                    "index": True,
                    "dimension": dimension
                }
            }
        }
        url = self._master_prefix + '/space/' + self._db_name + '/_create'
        response = requests.put(url, json=payload)
        print("create table",
              ", dimension: ", dimension,
              ", ncentroids: ", ncentroids,
              ", nprobe: ", nprobe,
              ", partition_num: ", partition_num,
              ", replica_num: ", replica_num,
              ", metric_type: ", metric_type,
              ", nsubvector: ", nsubvector)
        print("status: ", response.status_code)

    def _drop_table(self):
        url = self._master_prefix + '/space/' + self._db_name + '/' + self._table_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)

    def fit(self, X):
        dimension = X.shape[1]
        self._create_db()
        self._create_table(
            dimension,
            self._ncentroids,
            self._nprobe,
            self._nsubvector,
            self._partition_num,
            self._replica_num,
            self._metric_type,
        )
        # bulk insert
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_bulk'
        records_len = len(X)
        step = 20000
        for i in range(0, records_len, step):
            end = min(i + step, records_len)
            docs = ""
            for j in range(i, end):
                docs += json.dumps({
                    "index": {
                        "_id": j
                    }
                }) + "\n"
                docs += json.dumps({
                    self._field: {
                        "feature": X[j].tolist()
                    }
                }) + "\n"
            response = requests.request("POST", headers="Content-Type: application/json", data=docs)
            print("bulk insert docs: ", url, ", status: ", response.status_code)

    def batch_query(self, X, n):
        self._res = []
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_msearch'
        features = []
        for vector in X:
            features += vector
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n
        }
        response = requests.post(url, json=payload)
        print("query: ", url, ", status: ", response.status_code)

    def get_batch_results(self):
        return self._res

    def done(self):
        self._drop_table()
        self._drop_db()
        return
