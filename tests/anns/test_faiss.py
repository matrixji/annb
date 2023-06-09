import numpy as np
from annb.anns.faiss import FaissIndexUnderTest, FaissIndexUnderTestFactory
from annb.anns.indexes import MetricType

def test_faiss_index_under_test():
    index_under_test = FaissIndexUnderTest('IndexFlat', 4, MetricType.L2, index='flat')
    x = np.random.rand(100, 4).astype(np.float32)
    index_under_test.add(x)
    distances, ids = index_under_test.search(x[:3], 3)
    assert len(distances) == 3
    assert len(ids) == 3
    # the 1st column
    assert list(ids[:,0]) == [0, 1, 2]

def test_faiss_index_under_test_factory():
    factory = FaissIndexUnderTestFactory()
    index_under_test = factory.create('IndexFlat', 4, MetricType.L2, index='flat')
    x = np.random.rand(100, 4).astype(np.float32)
    index_under_test.add(x)
    assert index_under_test.index.ntotal == 100
