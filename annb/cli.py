from argparse import ArgumentParser
from typing import Union
from logging import getLogger, Formatter, StreamHandler, FileHandler
from sys import stdout

from .dataset.hdf5_dataset import AnnbHdf5Dataset
from .dataset.random_dataset import RandomDataset
from .indexes import IndexUnderTestFactory
from .plot import plot_result_recall_vs_qps
from .result import BenchmarkResult
from .runner import Runner
from .config import load_configs

logger = getLogger('annb')


def init_logger(log_level: str, log_file: Union[str, None] = None):
    """Initialize logger"""
    logger = getLogger('annb')
    log_level = log_level.upper()
    handler = StreamHandler(stdout)
    if log_file:
        handler = FileHandler(log_file)
    handler.setLevel(log_level)
    handler.setFormatter(
        Formatter(
            '%(asctime)s [%(name)s][%(levelname)s] %(filename)s:%(lineno)d, %(message)s'
        )
    )
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def load_dict(s):
    """
    >>> load_dict('a=1,b=2')
    {'a': 1, 'b': 2}
    """
    s = s.strip()
    data = {}
    for x in s.split(','):
        k, v = x.split('=')
        try:
            v = int(v)
        except ValueError:
            v = str(v)
        try:
            v = float(v)
        except ValueError:
            v = str(v)
    return data


def load_index_factory(index_factory, index_factory_args) -> IndexUnderTestFactory:
    """
    >>> load_index_factory('annb.anns.faiss.indexes.index_under_test_factory')
    <annb.anns.faiss.indexes.FaissIndexUnderTestFactory object at 0x7f6b3d0b9e10>
    """
    try:
        module_name, class_name = index_factory.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)(**index_factory_args)
    except Exception as e:
        logger.error('Load index factory failed', e)
        exit(1)


def create_or_load_dataset(dataset_file: str, dimension: int, metric_type: str):
    """
    >>> create_or_load_dataset('sift-128-euclidean.hdf5', 128, 'euclidean')
    <annb.dataset.hdf5_dataset.AnnbHdf5Dataset object at 0x7f6b3d0b9e10>
    """
    if dataset_file:
        try:
            dataset = AnnbHdf5Dataset(dataset_file)
        except Exception as e:
            logger.error('Load dataset failed', e)
            exit(1)
    else:
        temp_file = f'.random_d{dimension}_{metric_type}.hdf5'
        dataset = RandomDataset(temp_file, dimension=dimension, metric=metric_type)
    logger.info(
        'use dataset: %s, dim: %d, metric: %s',
        dataset,
        dataset.dimension,
        dataset.metric_type,
    )
    return dataset, dataset.dimension, dataset.metric_type


def run_once(
    name,
    index_factory,
    index_factory_args,
    index_name,
    index_dim,
    index_metric_type,
    index_args,
    query_args,
    dataset,
    result,
    topk,
    jobs,
    loop,
    step,
):
    factory = load_index_factory(index_factory, index_factory_args)
    dataset, index_dim, index_metric_type = create_or_load_dataset(
        dataset, index_dim, index_metric_type
    )
    index = factory.create(index_name, index_dim, index_metric_type, **index_args)
    logger.info(
        'use index: %s, dim: %d, metric: %s, %s',
        index,
        index.dimension,
        index.metric_type,
        index_args,
    )
    runner = Runner(
        name,
        index,
        dataset,
        query_args=query_args,
        topk=topk,
        jobs=jobs,
        loop=loop,
        step=step,
    )
    runner.run()
    if result:
        runner.benchmark_result.save(result)
    else:
        print(runner.benchmark_result)


def run_file(filename):
    runs = load_configs(filename)
    for run in runs:
        run_once(
            run['name'],
            run['index_factory'],
            run['index_factory_args'],
            run['index_name'],
            run['index_dim'],
            run['index_metric_type'],
            run['index_args'],
            run['query_args'],
            run['dataset'],
            run['result'],
            run['topk'],
            run['jobs'],
            run['loop'],
            run['step'],
        )


def report_plain(inputs, output):
    output_file = stdout
    if output:
        output_file = open(output, 'w')
    for input in inputs:
        result = BenchmarkResult.load(input)
        output_file.write(f'# result for {input}:\n')
        output_file.write(f'{result}\n')
    output_file.close()


def report_csv(inputs, output):
    output_file = stdout
    if output:
        output_file = open(output, 'w')
    output_file.write(','.join(BenchmarkResult.csv_header) + '\n')
    for input in inputs:
        result = BenchmarkResult.load(input)
        for line in result.csv_output_lines():
            output_file.write(','.join([f'"{x}"' for x in line]) + '\n')


def report_png(inputs, output, **kwargs):
    data = [BenchmarkResult.load(input) for input in inputs]
    plot_result_recall_vs_qps(data, output=output, **kwargs)


def report_main():
    parser = ArgumentParser()
    parser.add_argument('input', default=[], help='Input report files', nargs='+')
    parser.add_argument(
        '--format',
        default='plain',
        choices=['csv', 'plain', 'png'],
        help='Output format',
    )
    parser.add_argument(
        '--output', default='', help='Output file, if not set will print to stdout'
    )
    parser.add_argument(
        '--title',
        default='Recall vs QPS',
        help='Title for png report, only used when format is png',
    )
    parser.add_argument(
        '--subtitle',
        default='',
        help='Subtitle for png report, only used when format is png',
    )
    parser.add_argument(
        '--foot-notes',
        default='',
        help='Foot notes for png report, only used when format is png',
    )
    opts = parser.parse_args()
    if not opts.input:
        print('No input files provided')
        exit(1)
    if opts.format == 'plain':
        report_plain(opts.input, opts.output)
    elif opts.format == 'csv':
        report_csv(opts.input, opts.output)
    elif opts.format == 'png':
        report_png(
            opts.input,
            opts.output,
            title=opts.title,
            subtitle=opts.subtitle,
            foot_notes=opts.foot_notes,
        )


def test_main():
    parser = ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level',
    )
    parser.add_argument(
        '--log-file', default=None, help='Log file, if not specified, log to stdout'
    )
    parser.add_argument(
        '--run-file', default='', help='Run config file, if not set use config from cli'
    )
    # options for run
    parser.add_argument(
        '--name', default='Test', help='Run name, if not set use index name'
    )
    parser.add_argument(
        '--index-factory', default='annb.anns.faiss.indexes.index_under_test_factory'
    )
    parser.add_argument(
        '--index-factory-args',
        default={},
        type=load_dict,
        help='Index factory args, comma separated key=value',
    )
    parser.add_argument('--index-name', default='Test', help='Index name')
    parser.add_argument(
        '--index-dim',
        default=256,
        type=int,
        help='Index dimension, only used when generate random dataset',
    )
    parser.add_argument(
        '--index-metric-type',
        default='l2',
        help='Index metric type, only used when generate random dataset',
    )
    parser.add_argument(
        '--index-args',
        default={'index': 'ivfflat', 'nlist': 128},
        type=load_dict,
        help='Index args, comma separated key=value',
    )
    parser.add_argument(
        '--query-args',
        default=[{'nprobe': 1}],
        type=load_dict,
        action='append',
        help='Query args, comma separated key=value, set multiple times to run multiple queries',
    )
    parser.add_argument('--topk', default=10, type=int, help='topk')
    parser.add_argument(
        '--step', default=10, type=int, help='step size, also as batch size'
    )
    parser.add_argument(
        '--jobs',
        default=1,
        type=int,
        help='jobs, how many query jobs to run in parallel',
    )
    parser.add_argument(
        '--loop',
        default=5,
        type=int,
        help='loop, how many times to run the query, only use the best one',
    )
    parser.add_argument(
        '--dataset',
        default='',
        help='Dataset file, if not set will generate random dataset',
    )
    parser.add_argument(
        '--result', default='', help='Result file, if not set will print to stdout'
    )

    opts = parser.parse_args()
    init_logger(opts.log_level, opts.log_file)
    logger.debug('run with options: %s', opts)
    if opts.run_file:
        run_file(opts.run_file)
    else:
        run_once(
            opts.name,
            opts.index_factory,
            opts.index_factory_args,
            opts.index_name,
            opts.index_dim,
            opts.index_metric_type,
            opts.index_args,
            opts.query_args,
            opts.dataset,
            opts.result,
            opts.topk,
            opts.jobs,
            opts.loop,
            opts.step,
        )


if __name__ == '__main__':
    test_main()
