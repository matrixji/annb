from time import monotonic
from .anns.indexes import MetricType, IndexUnderTest, IndexUnderTestFactory
from .dataset import BaseDataset


class Runner:
    """
    Runner for run benchmarks
    """

    def __init__(self, dataset: BaseDataset, index_factory: IndexUnderTestFactory, **kwargs) -> None:
        self.dataset = dataset
        self.index_factory = index_factory
        self.index = None
        self.interations = kwargs.get('iterations', 10)
        self.timeout = kwargs.get('timeout', -1)
        self.started = 0.0
        self.run_count = 0
        self.name = kwargs.get('name', 'test')
        self.dimension = kwargs.get('dimension', 128)
        self.metric_type = MetricType.from_text(
            kwargs.get('metric_type', 'l2').lower())
        # remove keys for args conflict
        for key in ('name', 'dimension', 'metric_type'):
            if key in kwargs:
                del kwargs[key]
        self.kwargs = kwargs

        # test types
        self.test_types = kwargs.get('test_types', 'search').split(',')
        self.test_train = 'train' in self.test_types
        self.test_search = 'search' in self.test_types
        self.test_recall = 'recall' in self.test_types
        self.validate()
        self.durations = {}

    def validate(self):
        """
        validate runner configuration
        """
        if self.dimension != self.dataset.dimension:
            raise ValueError('Dimension mismatch: {} != {}'.format(
                self.dimension, self.dataset.dimension))
        if self.metric_type != self.dataset.metric_type:
            raise ValueError('Metric type mismatch: {} != {}'.format(
                self.metric_type, self.dataset.metric_type))

    def duration_execution(self, stage, func, *args, **kwargs):
        """
        Measure execution duration of function

        :param stage: Stage name
        :param func: Function to measure
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        :return: Function result
        """
        duration_stage = stage.split('#')[0]
        started = monotonic()
        res = func(*args, **kwargs)
        duration = monotonic() - started
        print('Stage: {}, duration: {:.3f}ms'.format(stage, duration * 1000.0))
        self.durations.setdefault(duration_stage, []).append(duration)
        return res

    def run(self):
        """
        Run benchmarks
        """
        self.started = monotonic()
        if not self.test_train:
            # means test recall or search
            self.index = self.index_factory.create(
                self.name, self.dimension, self.metric_type, **self.kwargs)
            self.index.train(self.dataset.data)
            self.index.add(self.dataset.data)
            self.index.warmup()

        while not self.should_stop():
            self.run_iteration()
        for key in self.durations:
            durations = sum(self.durations[key]) / len(self.durations[key])
            print('Stage average: {}, duration: {:.3f}ms'.format(key, durations * 1000.0))

    def run_iteration(self):
        self.run_count += 1
        nq = int(self.kwargs.get('nq', 10))
        topk = int(self.kwargs.get('topk', 10))

        if self.test_train:
            # recreate the index
            if self.index:
                self.index.cleanup()
            self.index = self.index_factory.create(
                self.name, self.dimension, self.metric_type, **self.kwargs)
            self.duration_execution(
                'train', self.index.train, self.dataset.data)
            self.index.add(self.dataset.data)
        else:
            self.duration_execution(
                f'search({self.metric_type.name}),nb={self.dataset.count},dim={self.dataset.dimension},query={nq},k={topk}#{self.run_count}',
                self.index.search, self.dataset.query_data[:nq], topk)

    def get_search_data(self, n):
        """
        Get search data for search benchmark
        """
        return self.dataset.data[:n]

    def should_stop(self):
        if self.timeout < 0:
            return self.run_count >= self.interations
        return monotonic() - self.started > self.timeout or self.run_count >= self.interations
