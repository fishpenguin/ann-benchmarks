#!/bin/bash

sleep 2s

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss

python run.py --dataset sift-128-euclidean --definitions algos.yaml.faiss_hnsw

python run.py --dataset sift-128-euclidean --definitions algos.yaml.milvus

python run.py --dataset sift-128-euclidean --definitions algos.yaml.sptag

python run.py --dataset sift-128-euclidean --definitions algos.yaml.ann-es

python run.py --dataset glove-100-angular --definitions algos.yaml.faiss.angular

python run.py --dataset glove-100-angular --definitions algos.yaml.faiss_hnsw.angular

python run.py --dataset glove-100-angular --definitions algos.yaml.milvus.angular

python run.py --dataset glove-100-angular --definitions algos.yaml.sptag.angular

python run.py --dataset glove-100-angular --definitions algos.yaml.ann-es.angular

