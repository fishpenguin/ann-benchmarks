#!/usr/bin/env python

import milvus
import numpy
import h5py

client = milvus.Milvus(host='localhost', port='19535', try_connect=False, pre_ping=False)

fn = 'oss-mishards/datasets/sift-10m-euclidean.hdf5'
datasets = h5py.File(fn, 'r')

X = numpy.array(datasets['train'])
dimension = X.shape[1]

table_name = "test_meta_and_nas"
collection_params = {
    "collection_name": table_name,
    "dimension": dimension,
    "index_file_size": 2048,
    "metric_type": milvus.MetricType.L2,
}

status, has_table = client.has_collection(table_name)
if has_table:
    print("already has table {}".format(table_name))
    exit()
client.create_collection(collection_params)

vector_ids = [id for id in range(len(X))]
records = X.tolist()
records_len = len(records)
step = 20000
for i in range(0, records_len, step):
    end = min(i + step, records_len)
    status, ids = client.insert(collection_name=table_name, records=records[i:end], ids=vector_ids[i:end])
    if not status.OK():
        raise Exception("Insert failed. {}".format(status))
client.flush([self._table_name])

index_type = getattr(milvus.IndexType, "IVF_FLAT")  # a bit hacky but works
status = client.create_index(table_name, index_type, params={"nlist": 8192})
if not status.OK():
    raise Exception("Create index failed. {}".format(status))

print("create index done ...")
client.close()
