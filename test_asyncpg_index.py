#!/usr/bin/env python

from __future__ import absolute_import
import psycopg2
from psycopg2 import extras
import asyncio
import asyncpg
from ann_benchmarks.algorithms.base import BaseANN

class AnalyticDB(BaseANN):
    def __init__(
        self,
        dataset,
        database='gpdb',
        user='mygpdb',
        password='mygpdb',
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        port=3432,
    ):
        self._database = database
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._conn = psycopg2.connect(
            database=self._database,
            user=self._user,
            password=self._password,
            host=self._host,
            port=self._port,
        )
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()
        self._table_name = dataset.replace('-', '_')

    def fit(self, X):
        exist_sql = "select count(*) from pg_class where relname = '{}'".format(self._table_name)
        self._cursor.execute(exist_sql)
        exist_rows = self._cursor.fetchall()
        assert len(exist_rows) == 1
        if exist_rows[0][0] != 0:
            return # already in database, skip to save time
        create_sql = "create table {} (id serial primary key, vector real[])".format(self._table_name)
        dimension = X.shape[1]
        index_sql = "create index on {} using ann(vector) with (dim={})".format(self._table_name, dimension)
        self._cursor.execute(create_sql)
        self._cursor.execute(index_sql)
        self._conn.commit()

        row_nums = len(X)
        step = 10000
        insert_sql = "insert into {}(id, vector) values %s".format(self._table_name)
        for i in range(0, row_nums, step):
            end = min(i + step, row_nums)
            rows = [(j, X[j].tolist()) for j in range(i, end)]
            psycopg2.extras.execute_values(self._cursor, insert_sql, rows)
            self._conn.commit()

    def query(self, v, n):
        query_sql = """select id
                       from {}
                       order by vector <-> array{}
                       limit {}
        """.format(self._table_name, v.tolist(), n)
        self._cursor.execute(query_sql)
        ids = self._cursor.fetchall()
        return ids

    def __str__(self):
        return 'AnalyticDB for PostgreSQL, machine: %s' % (self._host)

    def done(self):
        self._conn.close()

class AnalyticDBAsync(AnalyticDB):
    def __init__(
        self,
        dataset,
        database='gpdb',
        user='mygpdb',
        password='mygpdb',
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        port=3432,
    ):
        AnalyticDB.__init__(
            self,
            dataset,
            database,
            user,
            password,
            host,
            port,
        )
        self._el = asyncio.get_event_loop()
        db_setting = {
            'database': database,
            'user': user,
            'password': password,
            'host': host,
            'port': port,
        }
        self._db_pool = self._el.run_until_complete(asyncpg.create_pool(
            **db_setting
        ))

    def fit(self, X):
        exist_sql = "select count(*) from pg_class where relname = '{}'".format(self._table_name)
        self._cursor.execute(exist_sql)
        exist_rows = self._cursor.fetchall()
        assert len(exist_rows) == 1
        if exist_rows[0][0] != 0:
            return # already in database, skip to save time

        create_sql = "create table {} (id serial primary key, vector real[])".format(self._table_name)
        dimension = len(X[0])
        index_sql = "create index on {} using ann(vector) with (dim={})".format(self._table_name, dimension)
        self._cursor.execute(create_sql)
        self._cursor.execute(index_sql)
        self._conn.commit()

        async def insert_records():
            print("insert")

            async def batch_insert(records):
                async with self._db_pool.acquire() as conn:
                    print("aha")
                    status = await conn.copy_records_to_table(self._table_name, records=records)
                    print(status)

            row_nums = len(X)
            # step = 10000
            step = 10
            coros = []
            for i in range(0, row_nums, step):
                end = min(i + step, row_nums)
                rows = [(j, X[j]) for j in range(i, end)]
                coros.append(batch_insert(rows))
            await asyncio.gather(*coros)

        self._el.run_until_complete(insert_records())

    def batch_query(self, X, n):
        query_sql = "select id from {} order by vector <-> array$1 limit {}".format(self._table_name, n)
        # query_sql = "select * from {} where id >= $1 order by id limit {}".format(self._table_name, n)

        rows = [[0] * n for _ in range(len(X))]
        async def query_async():
            print("query")

            async def single_query(i):
                async with self._db_pool.acquire() as conn:
                    print("single query")
                    rows[i] = await conn.fetch(query_sql, X[i])
                    print("i: {}, rows[{}]: {}".format(i, i, rows[i]))

            coros = [single_query(i) for i in range(len(X))]
            await asyncio.gather(*coros)
            self._res = rows

        self._el.run_until_complete(query_async())

    def get_batch_results(self):
        batch_results = []
        for topk in self._res:
            print(topk)
            batch_results.append([t.get('id') for t in topk])
        return batch_results

    def __str__(self):
        return 'Asyncorization AnalyticDB for PostgreSQL'

def main():
    client = AnalyticDBAsync(
        dataset='test_async',
        database='postgres',
        user='zilliz',
        password='Fantast1c',
        host='gp-bp1m6ek4zwlu0254ro.gpdb.rds.aliyuncs.com',
        port=3432,
    )
    dimension = 4
    vector_nums = 30
    X = [[i] * dimension for i in range(vector_nums)]
    client.fit(X)
    qs = X
    client.batch_query(qs, vector_nums)
    ids = client.get_batch_results()
    print(ids)

if __name__ == "__main__":
    main()
