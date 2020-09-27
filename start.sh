#!/bin/bash

sift="sift-10m-euclidean"
deep="deep-10m-angular"

for dataset in ${sift} ${deep}
do
  for algo in "milvus-ivf-flat" "milvus-ivf-sq8" "milvus-hnsw"
  do
    start_time=$(date +%Y-%m-%d-%H:%M:%S)
    echo "[${start_time}] algo ${algo} on dataset ${dataset} running ... "
    python3 run.py --dataset ${dataset} --algorithm ${algo} --batch -k 50 --local &> "${algo}--${dataset}--${start_time}.log"
    end_time=$(date +%Y-%m-%d-%H:%M:%S)
    echo "[${end_time}] algo ${algo} on dataset ${dataset} done. "
  done
done

pq_sift_start_time=$(date +%Y-%m-%d-%H:%M:%S)
echo "[${pq_sift_start_time}] algo ivf-pq on dataset ${sift} running ... "
python3 run.py --dataset ${sift} --batch -k 50 --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-pq.sift &> "milvus-ivf-pq--${sift}--${pq_sift_start_time}.log"
pq_sift_end_time=$(date +%Y-%m-%d-%H:%M:%S)
echo "[${pq_sift_end_time}] algo ivf-pq on dataset ${sift} done."

pq_deep_start_time=$(date +%Y-%m-%d-%H:%M:%S)
echo "[${pq_deep_start_time}] algo ivf-pq on dataset ${deep} running ... "
python3 run.py --dataset ${deep} --batch -k 50 --local --definitions definitions/on-cloud/algos.yaml.milvus-ivf-pq.deep &> "milvus-ivf-pq--${deep}--${pq_deep_start_time}.log"
pq_deep_end_time=$(date +%Y-%m-%d-%H:%M:%S)
echo "[${pq_deep_end_time}] algo ivf-pq on dataset ${deep} done."
