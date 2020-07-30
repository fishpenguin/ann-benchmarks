#!/bin/bash

#python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.faiss-ivf.8192
#python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.faiss-ivf.4096
#python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.faiss-ivf.2048
#python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.faiss-ivf.1024
#python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.faiss-ivf.512

python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.milvus-ivf-flat.8192
python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.milvus-ivf-flat.4096
python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.milvus-ivf-flat.2048
python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.milvus-ivf-flat.1024
python run.py --dataset sift-128-euclidean --batch --definitions algos.yaml.milvus-ivf-flat.512
