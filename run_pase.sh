#!/bin/bash

python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.pase.hnsw
python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.pase.ivfflat

python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.pase.hnsw
python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.pase.ivfflat

python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.pase.hnsw
python run.py --dataset sift-100m-euclidean --batch --local --definitions definitions/on-cloud/algos.yaml.pase.ivfflat
