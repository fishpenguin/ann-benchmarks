#!/usr/bin/env python

import os

def main():
    all_files = os.listdir('.')
    for fn in all_files:
        if 'pq' in fn and 'faiss' in fn:
            with open(fn, 'r') as f:
                content = f.read().replace('ivf-pq', 'o-ivf-pq')
                content = content.replace('IVF', 'OIVF')
            with open(fn.replace('ivf-pq', 'o-ivf-pq'), 'w') as f:
                f.write(content)

if __name__ == "__main__":
    main()

