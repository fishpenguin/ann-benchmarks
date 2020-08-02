#!/bin/bash

sleep 12h

python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.hnsw

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf.512

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-flat.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-flat.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-flat.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-flat.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-flat.512

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-pq.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-pq.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-pq.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-pq.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-pq.512

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-pq.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-pq.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-pq.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-pq.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-pq.512

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.faiss-gpu-ivf-sq

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-sq8.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-sq8.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-sq8.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-sq8.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-gpu-ivf-sq8.512

python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-ivf-sq8h.8192
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-ivf-sq8h.4096
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-ivf-sq8h.2048
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-ivf-sq8h.1024
python run.py --dataset sift-128-euclidean --batch --local --definitions algos.yaml.milvus-ivf-sq8h.512
