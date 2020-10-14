import argparse
import math
import os
import time
import h5py
import numpy as np
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.algorithms import milvus_ivf_flat, milvus_hnsw


def task_mix(w_weight, r_weight):
    wr_pair = w_weight - r_weight if w_weight > r_weight else r_weight - w_weight
    wr_list = ['r', 'w'] * wr_pair

    if w_weight > r_weight:
        wr_list.extend(['w'] * (w_weight - r_weight))
    else:
        wr_list.extend(['r'] * (r_weight - w_weight))

    return wr_list


def store_result(records_points, dataset, algo, ww, rw):

    if not os.path.exists("payload_results"):
        os.mkdir("payload_results")

    dataset_dir = "payload_results/" + dataset
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    algo_dir = dataset_dir + "/" + algo
    if not os.path.exists(algo_dir):
        os.mkdir(algo_dir)

    ratio_dir = algo_dir + "/" + "{}_{}".format(rw, ww)
    if not os.path.exists(ratio_dir):
        os.mkdir(ratio_dir)

    for k, v in records_points.items():
        file_name = ratio_dir + "/" + str(k) + ".hdf5"
        with h5py.File(file_name, 'w') as f:
            f.attrs['rcount'] = v['rcount']
            f.attrs['rtime'] = v['rtime']
            f.attrs['wcount'] = v['wcount']
            f.attrs['wtime'] = v['wtime']

        print(f"Store {v} in file {file_name} done.")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        metavar='NAME',
        help='the dataset to load training points from',
        default='glove-100-angular',
        # choices=DATASETS.keys()
    )
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='run only the named algorithm',
        default=None)
    parser.add_argument(
        '--rweight',
        type=int,
        help="read request weight"
    )
    parser.add_argument(
        '--wweight',
        type=int,
        help="write request weight"
    )
    parser.add_argument(
        '--query-param',
        type=int,
        help="total test time. unit is minute",
        default=10
    )
    parser.add_argument(
        '--total-time',
        type=int,
        help="total test time. unit is minute",
        default=10
    )
    parser.add_argument(
        '--batch',
        type=int,
        help="request batch",
        default=1000
    )

    args = parser.parse_args()
    print("Run payload: "
          f" --dataset {args.dataset}"
          f" --algorithm {args.algorithm}"
          f" --rweight {args.rweight}"
          f" --wweight {args.wweight}"
          f" --query-param {args.query_param}"
          f" --total-time {args.total_time}"
          f" --batch {args.batch}\n", flush=True)

    print("Loading dataset ", args.dataset)
    dataset = get_dataset(args.dataset)
    X_train = np.array(dataset['train'])
    X_test = np.array(dataset['test'])
    metric = dataset.attrs['distance']
    dataset_name = args.dataset

    if args.algorithm == "IVF_FLAT":
        client = milvus_ivf_flat.MilvusIVFFLAT(metric, dataset_name, "IVF_FLAT", 4096)
    elif args.algorithm == "HNSW":
        client = milvus_hnsw.MilvusHNSW(metric, dataset_name, {"M": 64, "efConstruction": 500})
    else:
        raise ValueError("Unknown algorithm ", args.algorithm)

    total_fit = X_train.shape[0] - 100000
    # total_fit = X_train.shape[0] - 9000000
    fit_batch = 100000
    print("insert data ...")
    for offset in range(0, total_fit, fit_batch):
        print("{} / {} ...".format(offset, total_fit), flush=True)
        client.batch_fit(X_train[offset: offset + fit_batch], total_fit)

    X_insert = X_train[total_fit: total_fit + args.batch]
    X_query = X_test[: args.batch]

    client.set_query_arguments(args.query_param)
    tasks = task_mix(args.wweight, args.rweight)
    t0 = time.time()
    task_records = []
    while True:
        for task in tasks:
            ts = time.time()
            if task == 'r':
                client.batch_query(X_query, 10)
            elif task == 'w':
                client.batch_fit(X_insert, -1)
            record = {"end": time.time() - t0, "time": time.time() - ts, "task": task, "batch_size": args.batch}
            print("<{}>: ".format(time.time() - t0), record)
            task_records.append(record)
        if time.time() - t0 > args.total_time * 60:
            break

    print("store results")
    record_plot_points = {}
    for record in task_records:
        t = math.ceil(record['end'])
        tm = t // 60
        if tm not in record_plot_points:
            # record_plot_points[tm] = {"rcount": args.batch, "wcount": args.count}
            print(f"Start record minute {tm} data")
            record_plot_points[tm] = {"rcount": 0, "wcount": 0, "rtime": 0.0, "wtime": 0.0}
        if record["task"] == "r":
            record_plot_points[tm]["rcount"] += record["batch_size"]
            print(f'record rtime[{tm}] {record_plot_points[tm]["rtime"]} add {record["time"]}')
            record_plot_points[tm]["rtime"] += record["time"]
        else:
            record_plot_points[tm]["wcount"] += record["batch_size"]
            record_plot_points[tm]["wtime"] += record["time"]

    store_result(record_plot_points, dataset_name, args.algorithm, args.wweight, args.rweight)
    # ivfflat_client.batch_query(X_test, 10)
    client.done()


if __name__ == '__main__':
    main()

