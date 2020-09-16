from __future__ import absolute_import
import asyncio
import uuid
import time
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from ann_benchmarks.algorithms.base import BaseANN


class AliESAsyncHnsw(BaseANN):
    def __init__(self, metric, dataset, method_param):
        self._metric = metric
        self._index_name = dataset.replace('-', '_') + str(uuid.uuid1()).replace('-', '_')
        print("index name: ", self._index_name)
        self._method_param = method_param
        self._ef = None
        self._field = "vec"
        # self._es = Elasticsearch([ip], port=port)
        self._es = AsyncElasticsearch("http://es-cn-oew1t7131000y8424.elasticsearch.aliyuncs.com:9200",
                                      http_auth=("******", "********"), max_retries=5, retry_on_timeout=True, timeout=30)
        self._loop = asyncio.get_event_loop()

    def get_memory_usage(self):
        # a, b = loop.run_until_complete(asyncio.gather(*tasks))
        # exists = self._loop.run_until_complete(asyncio.gather(self._es.indices.exists(index=self._index_name)))
        exists = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if not exists:
            return 0
        # if not self._es.indices.exists(index=self._index_name):
        #     return 0

        stats = self._loop.run_until_complete(self._es.indices.stats(index=self._index_name))
        # stats = self._es.indices.stats(index=self._index_name)
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

        exist = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if exist:
            self._loop.run_until_complete(self._es.indices.delete(index=self._index_name))
        # create index mappings
        self._loop.run_until_complete(self._es.indices.create(index=self._index_name, body=_index_body))

        # bulk_records = []
        async def gen_action():
            for i, vec in enumerate(X.tolist()):
                # print("i: ", i, ", vec: ", vec)
                yield {
                    "_index": self._index_name,
                    "_id": i,
                    "_source": {
                        self._field: vec,
                    }
                }
                # bulk_records.append(doc_template)

        success, failed = self._loop.run_until_complete(
            async_bulk(self._es, gen_action(), index=self._index_name, raise_on_error=True))
        # success, _ = bulk(self._es, bulk_records, index=self._index_name, raise_on_error=True)
        self._loop.run_until_complete(self._es.indices.refresh(index=self._index_name))  # refresh to update index

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

        return self._es.search(index=self._index_name, body=query_params)
        # print("top_n_results: ", top_n_results)
        # top_n_results = [int(doc['_id']) for doc in top_n_results['hits']['hits']]
        # return top_n_results

    def handle_query_list_result(self, query_list):
        t0 = time.time()
        ql = (future for _, _,  future in query_list)
        results = self._loop.run_until_complete(asyncio.gather(*ql))
        handled_result = []

        for q, result in zip(query_list, results):
            total, v, _ = q
            result_ids = [int(doc['_id']) for doc in result['hits']['hits']]
            handled_result.append([total, v, result_ids])

        return time.time() - t0, handled_result

    def done(self):
        exists = self._loop.run_until_complete(self._es.indices.exists(index=self._index_name))
        if exists:
            print("delete index...")
            self._loop.run_until_complete(self._es.indices.delete(index=self._index_name))

        self._loop.run_until_complete(self._es.close())

    def __str__(self):
        return "Elasticsearch({}, param: {}, search param: {})".format("euclidean", self._method_param, self._ef)
