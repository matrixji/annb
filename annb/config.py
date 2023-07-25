"""
run config file use yaml format:

---
default:
  index_factory: <default index factory, default annb.anns.faiss.indexes.index_under_test_factory>
  index_factory_args: <the default index factory args, if not set use {}>
  index_name: <the default index name, if not set use Test>
  index_dim: <the default index dimension, if not set use from dataset>
  index_metric_type: <the default index metric type, if not set use from dataset>
  index_args: <the default index args, if not set use {}>
  query_args: <the default query args, if not set use {}>
  topk: <the default topk, if not set use 10>
  step: <the default step, if not set use 10>
  jobs: <the default jobs, if not set use 1>
  loop: <the default loop, if not set use 5>
  dataset: <the default dataset, if not set use annb.RandomDataset>
  result: <the default result file, if not set use None>

runs:
  -  name: <the run name, if not set use index name>
     index_factory: <the index factory, if not set use default>
  
"""

import yaml


def load_configs(filename: str):
    default_config = {
        'name': 'Test',
        'index_factory': 'annb.anns.faiss.indexes.index_under_test_factory',
        'index_factory_args': {},
        'index_name': 'Test',
        'index_dim': None,
        'index_metric_type': None,
        'index_args': {},
        'query_args': [[{'nprobe': 1}]],
        'topk': 10,
        'step': 10,
        'jobs': 1,
        'loop': 5,
        'dataset': 'annb.RandomDataset',
        'result': None,
    }
    data = {}
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if 'default' in data:
        default_config.update(data['default'])
    runs = []
    for run in data.get('runs', []):
        run_config = default_config.copy()
        run_config.update(run)
        runs.append(run_config)
    return runs
