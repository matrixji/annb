from collections.abc import Mapping
from queue import Empty
import sys
from collections import namedtuple
from logging import getLogger, Logger, DEBUG
from time import monotonic_ns
from multiprocessing.dummy import Process, Queue
from typing import List, Dict
from datetime import datetime

import numpy as np

from .indexes import IndexUnderTest
from .dataset import BaseDataset
from .result import BenchmarkResult

SingleResult = namedtuple(
    'SingleResult', ['distances', 'labels', 'time', 'test_indexes', 'count']
)

class RunnerLog(Logger):
    def __init__(self, name, rlog):
        super().__init__(name, DEBUG)
        self._logger = getLogger('annb')
        self.rlog = rlog

    def _log(self, level, msg, *args, **kwargs):
        self._logger._log(level, msg, *args, **kwargs)
        if self.rlog:
            self.rlog._log(level, msg, *args, **kwargs)

class Runner:
    def __init__(self, name, index: IndexUnderTest, dataset: BaseDataset, **kwargs):
        self.name = name
        self.index = index
        self.dataset = dataset
        self.query_args = kwargs.get('query_args', [])
        self.topk = kwargs.get('topk', 10)
        self.step = kwargs.get('step', 10)
        self.jobs = kwargs.get('jobs', 1)
        self.loop = kwargs.get('loop', 5)
        self.benchmark_result = BenchmarkResult()
        self.loop_index = 0
        self.queue = Queue()
        self.records = {}
        for key, value in kwargs.items():
            self.benchmark_result.add_attribute(key, value)
        self.benchmark_result.add_attribute('name', self.name)
        self.benchmark_result.add_attribute('topk', self.topk)
        self.benchmark_result.add_attribute('step', self.step)
        self.benchmark_result.add_attribute('jobs', self.jobs)
        self.benchmark_result.add_attribute('loop', self.loop)
        self.benchmark_result.add_attribute('query_args', self.query_args)
        self.benchmark_result.add_attribute('dataset', self.dataset.name)
        self.benchmark_result.add_attribute('index', self.index.name)
        self.benchmark_result.add_attribute('dim', self.index.dimension)
        self.benchmark_result.add_attribute('metric_type', self.index.metric_type)
        self.benchmark_result.add_attribute('index_args', self.index.kwargs)
        self.benchmark_result.add_attribute(
            'started', datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        if self.step == 0:
            # use all test data query once for batch mode
            self.step = self.dataset.test.shape[0]
        self.rlog = kwargs.get('rlog', None)
        self.log = RunnerLog(name, self.rlog)
        self.log.info('Runner init with %s', kwargs)

    def duration_run(self, text, func, *args, **kwargs):
        started = monotonic_ns()
        res = func(*args, **kwargs)
        duration = monotonic_ns() - started
        self.log.info('%s: %fms', text, duration / 1000000.0)
        return res, duration

    def run(self):
        self.index.cleanup()
        _, duration = self.duration_run(
            f'train {len(self.dataset.train)} items',
            self.index.train,
            self.dataset.train,
        )
        self.benchmark_result.add_training_duration(len(self.dataset.train), duration)
        _, duration = self.duration_run(
            f'add {len(self.dataset.data)} items', self.index.add, self.dataset.data
        )
        self.benchmark_result.add_insert_duration(len(self.dataset.data), duration)
        self.run_search_loop()

    def run_search_loop(self):
        self.index.warmup()
        query_args = self.query_args or [None]
        for i, query_arg in enumerate(query_args):
            if query_arg:
                if isinstance(query_arg, Dict):
                    self.index.update_search_args(**query_arg)
                    self.log.info('Update query args: %s', query_arg)
            self.records.clear()
            for loop_index in range(self.loop):
                self.loop_index = loop_index
                self.run_search()
            self.finalize_result(query_arg)
            self.log.info('Finish query args(%d/%d)', i + 1, len(query_args))

    @classmethod
    def run_multi_search(cls, args):
        for arg in args:
            cls.run_single_search(*arg)

    @classmethod
    def run_single_search(
        cls,
        index: IndexUnderTest,
        xq: np.array,
        test_indexes: List[int],
        topk: int,
        queue: Queue,
    ):
        assert len(xq) == len(test_indexes)
        start = monotonic_ns()
        distances, labels = index.search(xq, topk)
        end = monotonic_ns()
        result = SingleResult(distances, labels, end - start, test_indexes, len(xq))
        queue.put(result)

    def finalize_result(self, query_arg: Dict):
        best_loop = self.find_best_loop()
        best_results = self.records[best_loop]
        best_results = sorted(best_results, key=lambda r: r.test_indexes[0])
        np.concatenate([r.distances for r in best_results])
        labels = np.concatenate([r.labels for r in best_results])
        durations = [(len(r.test_indexes), r.time) for r in best_results]
        correct_count = 0
        total_count = 0
        ground_truth_neighbors = self.dataset.ground_truth_neighbors[:, : self.topk]

        # calc recall between ground_truth_neighbors and labels
        for i, (gt, test_items) in enumerate(zip(ground_truth_neighbors, labels)):
            total_count += len(gt)
            correct_count += len(set(gt) & set(test_items))
        recall = correct_count / total_count
        self.log.info('recall %.6f(%d/%d)', recall, correct_count, total_count)
        self.benchmark_result.add_query_result(
            recall=recall, durations=durations, query_arg=query_arg
        )

    def find_best_loop(self):
        # select which loop is the best
        best_loop = -1
        best_time = sys.maxsize
        for loop_index, records in self.records.items():
            time = 0
            indexes = []
            for record in records:
                time += record.time
                indexes.extend(record.test_indexes)
            indexes = sorted(indexes)
            if time < best_time and indexes == list(range(len(indexes))):
                best_loop = loop_index
                best_time = time
        if best_loop < 0:
            raise RuntimeError('No best loop found')
        self.log.info(
            '%s best loop: %d, with total duration: %fms',
            self.name,
            best_loop + 1,
            best_time / 1000000,
        )
        return best_loop

    def handle_result(self, result, proceed, total):
        self.log.debug(
            '%s single result: %d queries (%d/%d), %fms',
            self.name,
            result.count,
            proceed,
            total,
            result.time / 1000000,
        )
        self.records.setdefault(self.loop_index, []).append(result)

    def run_search(self):
        xq = self.dataset.test
        total_count = len(xq)
        jobs_args_list = {}
        for i in range(0, total_count, self.step):
            index = i // self.step % self.jobs
            indexes = list(range(i, min(i + self.step, total_count)))
            job_arg = (
                self.index,
                xq[i : i + self.step],
                indexes,
                self.topk,
                self.queue,
            )
            jobs_args_list.setdefault(index, []).append(job_arg)
        jobs = []
        for pargs in jobs_args_list.values():
            p = Process(target=self.run_multi_search, args=(pargs,))
            jobs.append(p)
            p.start()
        # collect the result
        for i in range(0, total_count, self.step):
            try:
                ret = self.queue.get(timeout=180)
            except Empty:
                self.log.error(
                    'Timeout when waiting for result, not result return in 180 seconds'
                )
                break
            if isinstance(ret, str):
                self.log.debug('debug from subprocess: %s', ret)
            else:
                proceed_count = i + self.step
                self.handle_result(ret, proceed_count, total_count)
        for p in jobs:
            p.join()
        self.log.info(
            'Finish %d queries in loop(%d/%d)',
            total_count,
            self.loop_index + 1,
            self.loop,
        )
