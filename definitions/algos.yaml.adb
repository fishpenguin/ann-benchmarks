float:
  euclidean:
    adb-async:
      docker-tag: ann-benchmarks-adb-async
      module: ann_benchmarks.algorithms.adb
      constructor: AnalyticDBAsync
      base-args: ["@dataset"]
      run-groups:
        base:
          args: [["gp-bp19e1yl22d85gqgxo-master.gpdbmaster.rds.aliyuncs.com"], [1, 2, 3]]
          query-args: [[1, 2, 3]]
