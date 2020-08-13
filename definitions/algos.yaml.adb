float:
  euclidean:
    adb-async:
      docker-tag: ann-benchmarks-adb-async
      module: ann_benchmarks.algorithms.adb
      constructor: AnalyticDBAsync
      base-args: ["@metric"]
      run-groups:
        base:
          args: [["@dataset"], ["gp-bp19e1yl22d85gqgx-master.gpdbmaster.rds.aliyuncs.com"], [1, 2, 3]]
          query-args: [[1, 2, 3]]
