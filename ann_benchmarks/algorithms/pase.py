from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.algorithms.adb import AnalyticDB, AnalyticDBAsync
import psycopg2
from psycopg2 import extras
import asyncio
import asyncpg

class PaseHNSWAsync(AnalyticDBAsync):
    def __init__(
        self,
        dataset,
        base_nb_num,
        ef_build,
        host='pgm-bp174758o17b9a8ylo.pg.rds.aliyuncs.com',
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=1921,
    ):
        AnalyticDBAsync.__init__(
            self,
            dataset,
            host,
            useless,
            database,
            user,
            password,
            port,
        )
        self._base_nb_num = base_nb_num
        self._ef_build = ef_build
        postfix = '_' + str(base_nb_num) + '_' + str(ef_build) + '_hnsw'
        self._table_name += postfix

    def _create_table(self):
        create_sql = "create table {} (id int, vector float4[])".format(self._table_name)
        print(create_sql)
        self._cursor.execute(create_sql)
        self._conn.commit()

    def _create_index(self, dimension):
        index_sql = """\
create index on {} \
using pase_hnsw(vector) \
with (dim = {}, base_nb_num = {}, ef_build = {}, ef_search = {}, base64_encoded = {})
""".format(self._table_name, dimension, self._base_nb_num, self._ef_build, 200, 0)
        print(index_sql)
        self._cursor.execute(index_sql)
        self._conn.commit()

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search

    def batch_query(self, X, n):
        rows = list(range(len(X)))
        async def query_async():
            async def single_query(i):
                async with self._db_pool.acquire() as conn:
                    query_data = ''
                    for j, c in enumerate(X[i]):
                        query_data += str(c) + ','
                    metric_type = '0' # euclidean distance
                    query_data = query_data[:-1] + ':' + str(self._ef_search) + ':' + metric_type
                    sql = "select id from {} order by vector <?> '{}'::pase limit {}".format(self._table_name, query_data, n)
                    rows[i] = await conn.fetch(sql)

            coros = [single_query(i) for i in range(len(X))]
            await asyncio.gather(*coros)
            self._res = rows

        self._el.run_until_complete(query_async())

    def __str__(self):
        return "pase hnsw, machine: {}, base_nb_num: {}, ef_build: {}, ef_search: {}".format(
            self._host,
            self._base_nb_num,
            self._ef_build,
            self._ef_search,
        )

class PaseIVFFLATAsync(AnalyticDBAsync):
    def __init__(
        self,
        dataset,
        k, # nlist
        host='pgm-bp174758o17b9a8ylo.pg.rds.aliyuncs.com',
        clustering_sample_ratio=100,
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=1921,
        clustering_type=1,
        distance_type=0, # euclidean distance
    ):
        AnalyticDBAsync.__init__(
            self,
            dataset,
            host,
            useless,
            database,
            user,
            password,
            port,
        )
        self._clustering_sample_ratio = clustering_sample_ratio
        self._k = k
        self._clustering_type = clustering_type
        self._distance_type = distance_type
        postfix = '_' + str(clustering_sample_ratio) + '_' + str(k) + '_' + str(clustering_type) + '_' + str(distance_type) + '_ivfflat'
        self._table_name += postfix

    def _create_table(self):
        create_sql = "create table {} (id int, vector float4[])".format(self._table_name)
        print(create_sql)
        self._cursor.execute(create_sql)
        self._conn.commit()

    def _create_index(self, dimension):
        index_sql = '''\
create index on {} \
using pase_ivfflat(vector) \
with (clustering_type = {}, distance_type = {}, dimension = {}, base64_encoded = {}, clustering_params = "{},{}")
'''.format(self._table_name, self._clustering_type, self._distance_type, dimension, 0, self._clustering_sample_ratio, self._k)
        print(index_sql)
        self._cursor.execute(index_sql)
        self._conn.commit()

    def set_query_arguments(self, nprobe):
        self._nprobe = nprobe

    def batch_query(self, X, n):
        rows = list(range(len(X)))
        async def query_async():
            async def single_query(i):
                async with self._db_pool.acquire() as conn:
                    query_data = ''
                    for j, c in enumerate(X[i]):
                        query_data += str(c) + ','
                    metric_type = '0' # euclidean distance
                    query_data = query_data[:-1] + ':' + str(self._nprobe) + ':' + metric_type
                    sql = "select id from {} order by vector <#> '{}' limit {}".format(self._table_name, query_data, n)
                    rows[i] = await conn.fetch(sql)

            coros = [single_query(i) for i in range(len(X))]
            await asyncio.gather(*coros)
            self._res = rows

        self._el.run_until_complete(query_async())

    def batch_query_sync(self, X, n):
        def single_query(i):
            query_data = ''
            for j, c in enumerate(X[i]):
                query_data += str(c) + ','
            metric_type = '0' # euclidean distance
            query_data = query_data[:-1] + ':' + str(self._nprobe) + ':' + metric_type
            sql = "select id from {} order by vector <#> '{}' limit {}".format(self._table_name, query_data, n)
            self._cursor.execute(sql)
            return self._cursor.fetchall()
        
        self._res = [single_query(i) for i in range(len(X))]

    def __str__(self):
        return "pase ivfflat, machine: {}, k: {}, nprobe: {}, clustering_sample_ratio: {}, distance_type: {}, clustering_type: {}".format(
            self._host,
            self._k,
            self._nprobe,
            self._clustering_sample_ratio,
            self._distance_type,
            self._clustering_type,
        )
