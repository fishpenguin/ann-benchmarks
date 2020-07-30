#!/bin/bash

docker build -t ann-benchmarks -f install/Dockerfile . --build-arg http_proxy=http://proxy.zilliz.tech:1087 --build-arg https_proxy=http://proxy.zilliz.tech:1087

docker build -t ann-benchmarks-milvus -f install/Dockerfile.milvus . --build-arg http_proxy=http://proxy.zilliz.tech:1087 --build-arg https_proxy=http://proxy.zilliz.tech:1087

docker build -t ann-benchmarks-faiss -f install/Dockerfile.faiss . --build-arg http_proxy=http://proxy.zilliz.tech:1087 --build-arg https_proxy=http://proxy.zilliz.tech:1087

docker build -t ann-benchmarks-sptag -f install/Dockerfile.sptag . --build-arg http_proxy=http://proxy.zilliz.tech:1087 --build-arg https_proxy=http://proxy.zilliz.tech:1087

docker build -t ann-benchmarks-ann-elasticsearch -f install/Dockerfile.ann-elasticsearch . --build-arg http_proxy=http://proxy.zilliz.tech:1087 --build-arg https_proxy=http://proxy.zilliz.tech:1087
