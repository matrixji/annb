from logging import INFO
from typing import List, Tuple
from numpy import ndarray
from annb.runner import Runner
from annb.anns.faiss.indexes import FaissIndexUnderTest
from annb.dataset import RandomDataset
from annb import MetricType



def test_abnormal_exit_on_sub_query(tmpdir):
    class DummyIndex(FaissIndexUnderTest):
        def __init__(self, index_name: str, dimension: int, metric_type: MetricType, **kwargs):
            super().__init__(index_name, dimension, metric_type, **kwargs)
            self.test_count = 0

        def search(self, query: ndarray, k: int) -> Tuple[List[float], List[int]]:
            if self.test_count == 7:
                raise RuntimeError('test')
            self.test_count += 1
            return super().search(query, k)
        
    with tmpdir.as_cwd():
        tmpdir.mkdir('cache')
        dataset = RandomDataset('cache/random_dataset.h5', metric='ip', dimension=4, count=2000, normalize=True)
        runner = Runner('test', DummyIndex(index_name='test', dimension=4, metric_type=MetricType.INNER_PRODUCT), dataset, jobs=5)
        runner.log.level = INFO
        runner.run_search()
        assert(len(runner.records) == 1)
        assert(len(runner.records[0]) == 7)
