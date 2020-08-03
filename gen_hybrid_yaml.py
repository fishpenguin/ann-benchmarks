#!/usr/bin/env python

import os

def main():
    all_files = os.listdir('.')
    for fn in all_files:
        if fn.startswith('algos.yaml.milvus-gpu'):
            with open(fn, 'r') as f:
                content = f.read().replace('milvus-gpu', 'milvus-gpu-hybrid')
            with open(fn.replace('milvus-gpu', 'milvus-gpu-hybrid'), 'w') as f:
                f.write(content)

if __name__ == "__main__":
    main()

