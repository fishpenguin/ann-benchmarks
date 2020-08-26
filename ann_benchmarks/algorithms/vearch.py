from __future__ import absolute_import
import json
import time
import numpy
import requests
from ann_benchmarks.algorithms.base import BaseANN

class Vearch(BaseANN):
    def __init__(self):
        self._db_name = 'annbench'
        self._table_name = 'annbench'
        self._field = 'field1'
        self._master_host = '172.16.0.251'
        self._master_port = '443'
        self._router_host = '172.16.0.251'
        self._router_port = '80'
        self._master_prefix = 'http://' + self._master_host + ':' + self._master_port
        self._router_prefix = 'http://' + self._router_host + ':' + self._router_port

    def _create_db(self):
        url = self._master_prefix + '/db/_create'
        response = requests.put(url, json={"name": self._db_name})
        print("post: ", url, ", status: ", response.status_code)

    def _drop_db(self):
        url = self._master_prefix + '/db/' + self._db_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)

    def _drop_table(self):
        url = self._master_prefix + '/space/' + self._db_name + '/' + self._table_name
        response = requests.delete(url)
        print("delete: ", url, ", status: ", response.status_code)

    def _create_table(self, dimension, retrieval_type, retrieval_param, partition_num=1, replica_num=1):
        payload = {
            "name": self._table_name,
            "partition_num": partition_num, # 数据分片数量，设为 PS 的数量比较合适
            "replica_num": replica_num, # 无副本，测量性能的话没必要高可用？
            "engine": {
                "name": "gamma",
                "retrieval_type": retrieval_type,
                "retrieval_param": retrieval_param
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
        print("create table: ", url)
        for key, value in retrieval_param.items():
            print(key, ": ", value)
        print("status: ", response.status_code)

    def _bulk_insert(self, X):
        dimension = X.shape[1]
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
            response = requests.request("POST", url, headers={"Content-Type": "application/json"}, data=docs)
            print("bulk insert docs: ", url, ", status: ", response.status_code)

    def batch_query(self, X, n):
        self._res = []
        url = self._router_prefix + '/' + self._db_name + '/' + self._table_name + '/_msearch'
        features = []
        for vector in X:
            features += vector.tolist()
        payload = {
            "query": {
                "sum": [{
                    "field": self._field,
                    "feature": features,
                }]
            },
            "size": n,
            "sort": [{
                "_score": {"order": "asc"}
            }]
        }
        if hasattr(self, "_nprobe"):
            payload["nprobe"] = self._nprobe
        response = requests.post(url, json=payload)
        print("query: ", url, ", status: ", response.status_code)
        if response.json():
            self._res = [[int(hit['_id']) for hit in results['hits']['hits']]
                         for results in response.json()['results']]

    def get_batch_results(self):
        return self._res

    def done(self):
        self._drop_table()
        self._drop_db()
        return

    def __str__(self):
        return "Vearch"

class VearchIVFPQ(Vearch):
    def __init__(self, ncentroids, nsubvector=64, partition_num=1, replica_num=1, metric_type='L2', nbits_per_idx=8):
        Vearch.__init__(self)
        self._ncentroids = ncentroids
        self._nsubvector = 64
        self._partition_num = partition_num
        self._replica_num = replica_num
        self._metric_type = metric_type
        self._nbits_per_idx = nbits_per_idx

    def fit(self, X):
        self._create_db()
        dimension = X.shape[1]
        retrieval_param = {
            "ncentroids": self._ncentroids,
            "nsubvector": self._nsubvector,
            "metric_type": self._metric_type,
            "nbits_per_idx": self._nbits_per_idx,
        }
        self._create_table(dimension, "IVFPQ", retrieval_param, self._partition_num, self._replica_num)
        self._bulk_insert(X)

    def set_query_arguments(self, nprobe):
        self._nprobe = nprobe

    def __str__(self):
        return ("VearchIVFPQ" +
                ", master: " + self._master_prefix +
                ", router: " + self._router_prefix +
                ", ncentroids: " + str(self._ncentroids) +
                ", nsubvector: " + str(self._nsubvector) +
                ", partition_num: " + str(self._partition_num) +
                ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type) +
                ", nbits_per_idx: " + str(self._nbits_per_idx))

class VearchHNSW(Vearch):
    def __init__(self, nlinks, efConstruction, efSearch, partition_num=1, replica_num=1, metric_type='L2'):
        Vearch.__init__(self)
        self._partition_num = partition_num
        self._replica_num = replica_num
        self._metric_type = metric_type
        self._nlinks = nlinks
        self._efConstruction = efConstruction
        self._efSearch = efSearch

    def fit(self, X):
        self._create_db()
        dimension = X.shape[1]
        retrieval_param = {
            "metric_type": self._metric_type,
            "nlinks": self._nlinks,
            "efConstruction": self._efConstruction,
            "efSearch": self._efSearch
        }
        self._create_table(dimension, "HNSW", retrieval_param, self._partition_num, self._replica_num)
        self._bulk_insert(X)

    def __str__(self):
        return ("VearchHNSW" +
                ", master: " + self._master_prefix +
                ", router: " + self._router_prefix +
                ", nlinks: " + str(self._nlinks) +
                ", efConstruction: " + str(self._efConstruction) +
                ", efSearch: " + str(self._efSearch) +
                ", partition_num: " + str(self._partition_num) +
                ", replica_num: " + str(self._replica_num) +
                ", metric_type: " + str(self._metric_type))
