from __future__ import absolute_import
import uuid
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from ann_benchmarks.algorithms.base import BaseANN


class AliESHNSW(BaseANN):
    def __init__(self, metric, dataset, method_param):
        self._metric = metric
        self._index_name = dataset.replace('-', '_') + str(uuid.uuid1()).replace('-', '_')
        print("index name: ", self._index_name)
        self._method_param = method_param
        self._ef = None
        self._field = "vec"
        # self._es = Elasticsearch([ip], port=port)
        self._es = Elasticsearch("http://es-cn-oew1t7131000y8424.public.elasticsearch.aliyuncs.com:9200",
                                 http_auth=("elastic", "Zilliz1314"), max_retries=5, retry_on_timeou=True, timeout=30)

    def get_memory_usage(self):
        if not self._es.indices.exists(index=self._index_name):
            return 0

        stats = self._es.indices.stats(index=self._index_name)
        return int(stats["_all"]["total"]["store"]["size_in_bytes"]) / 1024

    def fit(self, X):
        dim = X.shape[1]
        # print("dims: ", dims)
        _index_body = {
            "mappings": {
                "properties": {
                    self._field: {
                        "type": "proxima_vector",
                        "dim": dim
                    }
                }
            },
            "settings": {
                "index.codec": "proxima",
                "index.vector.algorithm": "hnsw",
                "index.vector.hnsw.builder.neighbor_cnt": self._method_param["M"] * 2,
                "index.vector.hnsw.builder.upper_neighbor_cnt": self._method_param["M"],
                "index.vector.hnsw.builder.efconstruction": self._method_param["efConstruction"],
                "index.refresh_interval": "1s",
                "index.number_of_replicas": 1,
                "index.number_of_shards": 1
            }
        }

        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)
        # create index mappings
        self._es.indices.create(index=self._index_name, body=_index_body)
        bulk_records = []
        for i, vec in enumerate(X.tolist()):
            # print("i: ", i, ", vec: ", vec)
            doc_template = {
                "_index": self._index_name,
                "_id": i,
                "_source": {
                    self._field: vec,
                }
            }
            bulk_records.append(doc_template)
        success, _ = bulk(self._es, bulk_records, index=self._index_name, raise_on_error=True)
        self._es.indices.refresh(index=self._index_name)  # refresh to update index

    def set_query_arguments(self, ef):
        self._ef = ef

    def query(self, v, n):
        query_params = {
            "query": {
                "hnsw": {
                    "vec": {
                        "vector": v,
                        "size": n,
                        "ef": self._ef
                    }
                }
            },
            "_source": False
        }

        top_n_results = self._es.search(index=self._index_name, body=query_params)
        # print("top_n_results: ", top_n_results)
        top_n_results = [int(doc['_id']) for doc in top_n_results['hits']['hits']]
        return top_n_results

    def done(self):
        if self._es.indices.exists(index=self._index_name):
            print("delete index...")
            self._es.indices.delete(index=self._index_name)

    def __str__(self):
        return ("Elasticsearch(%s, %s)" % ("euclidean", self._index_name))
