#!/bin/bash

python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.hnsw.milvus

# python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.8192 # ok
# python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.4096 # ok
# python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.2048 # ok 
# python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.1024 # ok
# python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.512  # ok 

# python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/algos.yaml.hnsw.milvus # ok

# python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.8192
python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.4096
python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.2048
python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.1024
python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-sq8.512
 
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.hnsw.milvus
#  
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.8192
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.4096
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.2048
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.1024
# python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/algos.yaml.milvus-ivf-sq8.512
