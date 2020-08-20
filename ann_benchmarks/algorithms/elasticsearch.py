from __future__ import absolute_import
import uuid
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from ann_benchmarks.algorithms.base import BaseANN

class ES(BaseANN):
    def __init__(self, metric, count):
        self._index_name = "ann_bench_" + str(uuid.uuid1()).replace('-', '_')
        print("index name: ", self._index_name)
        self._field = "vec"
        # self._es = Elasticsearch([ip], port=port)
        self._es = Elasticsearch()
    
    def fit(self, X):
        dims = X.shape[1]
        # print("dims: ", dims)
        _index_mappings = {
            "mappings": {
                "properties": {
                    self._field: {
                        "type": "dense_vector",
                        "dims": dims
                    }
                }
            }
        }
        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)
        # create index mappings
        self._es.indices.create(index=self._index_name, body=_index_mappings)
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
        self._es.indices.refresh(index=self._index_name) # refresh to update index

    def query(self, v, n):
        query_params = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        # TODO: more distance
                        "source": "1 / (1 + l2norm(params.queryVector, '%s'))" % (self._field),
                        "params": {
                            "queryVector": v.tolist()
                        }
                    }
                }
            },
            "from": 0,
            "size": n
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
