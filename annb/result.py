from typing import List, Dict
import os
import pickle

from annb.indexes import MetricType


class DurationWithCount:
    def __init__(self, count: int, duration: int):
        self.count = count
        self.duration = duration


class QueryResult:
    def __init__(self, recall: float, durations: List[DurationWithCount], args):
        self.recall = recall
        self.durations = durations
        self.args = args


class BenchmarkResult:
    csv_header = (
        'Test Name',
        'Started',
        'Index Name',
        'Index Dim',
        'Index Metric Type',
        'Index Args',
        'Topk',
        'Step',
        'Jobs',
        'Loop',
        'Dataset',
        'Training Durations(ms)',
        'Insert Durations(ms)',
        'Query Args',
        'Recall',
        'QPS',
        'Query Latency(ms)',
        'Query Latency P95(ms)',
        'Query Latency P99(ms)',
    )

    def __init__(self):
        self.training_durations = []
        self.insert_durations = []
        self.query_results = []
        self.attributes = {}

    def add_training_duration(self, count, duration):
        self.training_durations.append(DurationWithCount(count, duration))

    def add_insert_duration(self, count, duration):
        self.insert_durations.append(DurationWithCount(count, duration))

    def add_query_result(self, recall, durations: List, query_arg: Dict):
        result = QueryResult(recall, [], query_arg)
        for count, duration in durations:
            result.durations.append(DurationWithCount(count, duration))
        self.query_results.append(result)

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def csv_output_lines(self):
        for query_result in self.query_results:
            name = self.attributes.get('name', 'Test')
            started = self.attributes.get('started', 'NONE')
            index_name = self.attributes.get('index', 'Test')
            index_dim = self.attributes.get('dim', 0)
            index_metric_type = self.attributes.get('metric_type', 'Unknown')
            if isinstance(index_metric_type, MetricType):
                index_metric_type = index_metric_type.name
            index_args = self.attributes.get('index_args', {})
            topk = self.attributes.get('topk', 10)
            step = self.attributes.get('step', 10)
            jobs = self.attributes.get('jobs', 1)
            loop = self.attributes.get('loop', 5)
            dataset = self.attributes.get('dataset', 'Unknown')
            training_durations = sum([d.duration for d in self.training_durations])
            insert_durations = sum([d.duration for d in self.insert_durations])
            yield (
                name,
                started,
                index_name,
                str(index_dim),
                index_metric_type,
                str(index_args),
                str(topk),
                str(step),
                str(jobs),
                str(loop),
                dataset,
                str(training_durations / 1000000.0),
                str(insert_durations / 1000000.0),
                str(query_result.args),
                str(query_result.recall),
                str(BenchmarkResult.qps(query_result.durations)),
                str(BenchmarkResult.latency(query_result.durations) / 1000000.0),
                str(BenchmarkResult.latency_pn(query_result.durations, 95) / 1000000.0),
                str(BenchmarkResult.latency_pn(query_result.durations, 99) / 1000000.0),
            )

    @staticmethod
    def qps(durations):
        return sum([d.count for d in durations]) / (
            sum([d.duration for d in durations]) / 1000000000.0
        )

    @staticmethod
    def latency(durations):
        return sum([d.duration for d in durations]) / len(durations)

    @staticmethod
    def latency_pn(durations, n):
        duration_values = [d.duration for d in durations]
        duration_values.sort()
        return duration_values[int(len(duration_values) * n / 100)]

    @staticmethod
    def _summary(durations):
        return (
            f'{len(durations)} items,'
            f' {sum([d.count for d in durations])} total,'
            f' {sum([d.duration for d in durations])/1000000.0}ms'
        )

    @property
    def training_durations_summary(self):
        return self._summary(self.training_durations)

    @property
    def insert_durations_summary(self):
        return self._summary(self.insert_durations)

    def save(self, filename: str):
        abs_filepath = os.path.abspath(filename)
        os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)
        with open(abs_filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        attributes = ''
        for key, value in self.attributes.items():
            attributes += f'    {key}: {value}\n'
        query_durations = ''
        for query_result in self.query_results:
            args = ''
            if isinstance(query_result.args, Dict):
                for key, value in query_result.args.items():
                    args += f'{key}={value}'
            else:
                args = 'none'
            recall = f'recall={query_result.recall}'
            durations_total_query = sum([d.count for d in query_result.durations])
            durations_total_duration = sum([d.duration for d in query_result.durations])
            durations_total_qps = BenchmarkResult.qps(query_result.durations)
            latency = BenchmarkResult.latency(query_result.durations)
            latency_p95 = BenchmarkResult.latency_pn(query_result.durations, 95)
            latency_p99 = BenchmarkResult.latency_pn(query_result.durations, 99)
            duration_summary = f'{durations_total_query} items,'
            duration_summary += f' {durations_total_duration/1000000.0}ms,'
            duration_summary += f' {durations_total_qps}qps,'
            duration_summary += f' latency={latency/1000000.0}ms,'
            duration_summary += f' p95={latency_p95/1000000.0}ms,'
            duration_summary += f' p99={latency_p99/1000000.0}ms'
            query_durations += f'      {args},{recall} -> {duration_summary}\n'

        return f"""
BenchmarkResult:
  attributes:
{attributes}
  durations:
    training: {self.training_durations_summary}
    insert: {self.insert_durations_summary}
    query:
{query_durations}
"""
