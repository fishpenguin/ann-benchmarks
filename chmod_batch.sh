#!/bin/bash

cd results/sift-128-euclidean/10/faiss-ivf-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/faiss-ivf-pq-32-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/faiss-ivf-pq-64-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/milvus-ivf-flat-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/milvus-ivf-pq-32-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/milvus-ivf-pq-64-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/faiss-ivf-sq-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/milvus-ivf-sq8-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/faiss-sq-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/faiss-hnsw-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/milvus-hnsw-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/hnswlib-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -

cd results/sift-128-euclidean/10/sptag-batch
for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
cd -
