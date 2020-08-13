from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
import psycopg2
from psycopg2 import extras
import asyncio
import asyncpg

class AnalyticDB(BaseANN):
    def __init__(
        self,
        dataset,
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=5432,
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
            count_sql = "select count(*) from {}".format(self._table_name)
            self._cursor.execute(count_sql)
            count_rows = self._cursor.fetchall()
            if count_rows[0][0] >= len(X):
                return # already in database, skip to save time
            else:
                self._cursor.execute('drop table {}'.format(self._table_name))
                self._conn.commit()
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

    def set_query_arguments(self, useless='useless'):
        pass

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
        host='mygpdbpub.gpdb.rds.aliyuncs.com',
        useless='useless',
        database='postgres',
        user='annbench',
        password='Fantast1c',
        port=5432,
    ):
        AnalyticDB.__init__(
            self,
            dataset,
            host,
            useless,
            database,
            user,
            password,
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
            count_sql = "select count(*) from {}".format(self._table_name)
            self._cursor.execute(count_sql)
            count_rows = self._cursor.fetchall()
            if count_rows[0][0] >= len(X):
                print('already in database, skip to save time...')
                return # already in database, skip to save time
            else:
                print('drop table...')
                self._cursor.execute('drop table {}'.format(self._table_name))
                self._conn.commit()

        create_sql = "create table {} (id serial primary key, vector real[])".format(self._table_name)
        dimension = X.shape[1]
        index_sql = "create index on {} using ann(vector) with (dim={})".format(self._table_name, dimension)
        self._cursor.execute(create_sql)
        self._cursor.execute(index_sql)
        self._conn.commit()

        async def insert_records():
            async def batch_insert(records):
                async with self._db_pool.acquire() as conn:
                    await conn.copy_records_to_table(self._table_name, records=records)

            row_nums = len(X)
            step = 10000
            coros = []
            for i in range(0, row_nums, step):
                end = min(i + step, row_nums)
                rows = [(j, X[j].tolist()) for j in range(i, end)]
                coros.append(batch_insert(rows))
            await asyncio.gather(*coros)

        self._el.run_until_complete(insert_records())

    def batch_query(self, X, n):
        # query_sql = "select id from {} order by vector <-> array$1 limit {}".format(self._table_name, n)

        rows = list(range(len(X)))
        async def query_async():
            async def single_query(i):
                async with self._db_pool.acquire() as conn:
                    sql = "select id from {} order by vector <-> array{} limit {}".format(self._table_name, X[i].tolist(), n)
                    rows[i] = await conn.fetch(sql)
                    # rows[i] = await conn.fetch(query_sql, X[i].tolist())

            coros = [single_query(i) for i in range(len(X))]
            await asyncio.gather(*coros)
            self._res = rows

        self._el.run_until_complete(query_async())

    def get_batch_results(self):
        batch_results = []
        for topk in self._res:
            batch_results.append([t.get('id') for t in topk])
        return batch_results

    def __str__(self):
        return 'Asyncorization AnalyticDB for PostgreSQL, machine: {}'.format(self._host)
