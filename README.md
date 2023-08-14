# ANNB: Approximate Nearest Neighbor Benchmark

[![PyPI Version](https://img.shields.io/pypi/v/annb.svg)](https://pypi.python.org/pypi/annb)

Note: This is a work in progress. The API/CLI is not stable yet.

## Installation

```bash
pip install annb

# install vector search index/client you may need for benchmark
# e.g install faiss for run faiss index benchmark
```

## Usage

### CLI Usage

#### Run Benchmark

##### start first benchmark with a randome dataset.

Just run `annb-test` to start your first benchmark with a random dataset.

```bash
annb-test
```

It will produce a result like this:

```plain
❯ annb-test
... some logs ...

BenchmarkResult:
  attributes:
    query_args: [{'nprobe': 1}]
    topk: 10
    jobs: 1
    loop: 5
    step: 10
    name: Test
    dataset: .annb_random_d256_l2_1000.hdf5
    index: Test
    dim: 256
    metric_type: MetricType.L2
    index_args: {'index': 'ivfflat', 'nlist': 128}
    started: 2023-08-14 13:03:40

  durations:
    training: 1 items, 1000 total, 1490.03266ms
    insert: 1 items, 1000 total, 132.439627ms
    query:
      nprobe=1,recall=0.2173 -> 1000 items, 18.615083ms, 53719.878659686874qps, latency=0.18615083ms, p95=0.31939ms, p99=0.41488ms
```

This is a simple benchmark test with default index(faiss) with random l2 dataset.
If you wants to generate more data or with some different specifications for the dataset, you could see below options:
  - --index-dim         The dimension of the index, default is 256
  - --index-metric-type   Index metric type, l2 or ip, default is l2
  - --topk TOPK           topk used for query, default is 10
  - --step STEP           the query step, default annb will query 10 items per query, you could set it to 0 for query all items in one query (similar like batch for ann-benchmarks)
  - --batch               batch mode, alias --step 0
  - --count COUNT         the total number of items in the dataset, default is 1000

##### run benchmark with a specific dataset

You could also use ann-benchmarks's [dataset](https://github.com/erikbern/ann-benchmarks#data-sets) to run benchmark. download them locally and run benchmark with `--dataset` option.

```bash
annb-test --dataset sift-128-euclidean.hdf5
```

##### run benchmark with query args
You mary benchmark with different query args, e.g. different nprobe for faiss ivfflat index. you could try `--query-args` option.

```bash
annb-test --query-args nprobe=10 --query-args nprobe=20
```

will output below result:

```plain
durations:
    training: 1 items, 1000 total, 1548.84968ms
    insert: 1 items, 1000 total, 143.402532ms
    query:
      nprobe=1,recall=0.2173 -> 1000 items, 20.074236ms, 49815.09632545916qps, latency=0.20074235999999998ms, p95=0.332276ms, p99=0.455525ms
      nprobe=10,recall=0.5221 -> 1000 items, 49.141931ms, 20349.2207092961qps, latency=0.49141931ms, p95=0.722628ms, p99=0.818012ms
      nprobe=20,recall=0.6861 -> 1000 items, 69.284072ms, 14433.331805324606qps, latency=0.69284072ms, p95=1.126946ms, p99=1.350359ms
```

##### run multiple benchmarks with config file
You may run multiple benchmarks with different index and dataset. you could use `--run-file` run benchmarks from a config file.

Below is a example config file:

config.yaml

```yaml
default:
  index_factory: annb.anns.faiss.indexes.index_under_test_factory
  index_factory_args: {}
  index_name: Test
  dataset: gist-960-euclidean.hdf5
  topk: 10
  step: 10
  jobs: 1
  loop: 2
  result: output.pth

runs:
  - name: faiss-gist960-gpu-ivfflat
    index_args:
      gpu: yes
      index: ivfflat
      nlist: 1024
    query_args:
      - nprobe: 1
      - nprobe: 16
      - nprobe: 256
  - name: faiss-gist960-gpu-ivfpq8
    index_args:
      gpu: yes
      index: ivfpq
      nlist: 1024
    query_args:
      - nprobe: 1
      - nprobe: 16
      - nprobe: 256
```

Explanation for above config file:
- The default section is the default config for all benchmarks.
- The config keys are generally same as the options for `annb-test` command. e.g. `index_factory` is same as `--index-factory`.
- You could define multiple benchmarks in `runs` section. and each run config will override the default config. In this example, we define use gist-960-euclidean.hdf5 as dataset, so it will use this dataset for all benchmarks. and we use different index and query args for each benchmark. for index_args, we use ivfflat(nlist=1024) and ivfpq(nlist=1024) as two benchmark series. and for query_args, we use nprobe=1,16,256 for each benchmark. That means we will run 6 benchmarks in total, each series will run 3 benchmarks with different nprobe.
- The result will be saved to output.pth file by default setting. Actually, each benchmark series will save to a separate file. so in this example, we will get two files: `output-1.pth` and `output-2.pth`. you could use `annb-report` to view them.


##### more options

You could use `annb-test --help` to see more options.

```bash
❯ annb-test --help
```


#### Check Benchmark Results

The `annb-report` is use to view benchmark results as plain/csv text, or export them to Chart graphic.

```bash
annb-report --help
```

##### examples for view/export benchmark results

view benchmark results as plain text

```bash
annb-report output.pth
```

view benchmark results as csv text

```bash
annb-report output.pth --format csv
```

export benchmark results to chart graphic(multiple series)

```bash
annb-report output.pth --format png --output output.png output-1.pth output-2.pth
```
