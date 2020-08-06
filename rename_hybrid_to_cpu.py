#!/usr/bin/env python

import os
import h5py

BASE_DIR = 'results/sift-128-euclidean/10'

def main():
    algo_set = os.listdir(BASE_DIR)
    for algo in algo_set:
        if 'cpu' in algo:
            real_path = os.path.join(BASE_DIR, algo)
            res_set = os.listdir(real_path)
            for res_fn in res_set:
                f = h5py.File(os.path.join(real_path, res_fn), 'a')
                original_algo = str(f.attrs['algo'])
                cpu_algo = original_algo.replace('gpu-hybrid', 'cpu')
                f.attrs['algo'] = cpu_algo
                f.close()

                # test if code above works
                f = h5py.File(os.path.join(real_path, res_fn), 'r')
                print('algo: ', str(f.attrs['algo']))
                f.close()

if __name__ == "__main__":
    main()

