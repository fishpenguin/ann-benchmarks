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
