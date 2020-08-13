#!/bin/bash

python run.py --dataset sift-128-euclidean --batch --local --definitions definitions/algos.yaml.adb
python run.py --dataset gist-960-euclidean --batch --local --definitions definitions/algos.yaml.adb
