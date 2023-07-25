from unittest import mock
import numpy as np
from annb.dataset import utils
from annb.indexes import MetricType


def test_generate_groundtruth():
    data = np.random.rand(1000, 128).astype('float32')
    query = data
    metric_type = MetricType.L2
    ground_truth = utils.generate_groundtruth(query, data, metric_type)
    _, ids = ground_truth
    assert ids.shape == (1000, 100)
    assert ids.dtype == np.int64
    assert np.all(ids >= 0)
    assert np.all(ids < 1000)
    assert np.equal(ids[:, 0], np.arange(1000)).all()


def test_generate_groundtruth_without_faiss():
    # mock for faiss to remove knn_gpu function
    with mock.patch.dict('sys.modules', {'faiss': None}):
        test_generate_groundtruth()
