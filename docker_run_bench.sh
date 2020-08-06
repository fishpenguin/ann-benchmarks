#!/bin/bash

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf.512

python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-flat.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-flat.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-flat.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-flat.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-flat.512

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-pq.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-pq.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-pq.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-pq.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-pq.512

python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-pq.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-pq.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-pq.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-pq.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-pq.512

python run.py --dataset sift-128-euclidean --definitions algos.yaml.hnsw

python run.py --dataset sift-128-euclidean --definitions algos.yaml.sptag

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-ivf-sq.512

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss-sq

python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-sq8.8192
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-sq8.4096
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-sq8.2048
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-sq8.1024
python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus-ivf-sq8.512
