float:
  euclidean:
    adb-async:
      docker-tag: ann-benchmarks-adb-async
      module: ann_benchmarks.algorithms.adb
      constructor: AnalyticDBAsync
      base-args: ["@metric"]
      run-groups:
        base:
          args: [["@dataset"]]
