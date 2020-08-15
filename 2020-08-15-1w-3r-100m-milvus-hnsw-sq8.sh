#!/bin/bash

python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.hnsw.milvus
 
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.8192
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.4096
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.2048
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.1024
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.512

python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.hnsw.milvus
 
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.8192
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.4096
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.2048
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.1024
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.512
