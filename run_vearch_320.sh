#!/bin/bash

# test, warm up, save time
# python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.hnsw.320
# python run.py --dataset sift-10000-10 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfpq.320

python run.py --dataset sift-1000000-10000 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.hnsw.320
python run.py --dataset sift-1000000-10000 --batch --local --definitions definitions/on-cloud/algos.yaml.vearch.ivfpq.320
