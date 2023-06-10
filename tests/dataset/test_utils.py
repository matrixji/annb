from unittest import mock
import numpy as np
from annb.dataset import utils
from annb.anns.indexes import MetricType

def test_generate_groundtruth():
    data = np.random.rand(1000, 128).astype('float32')
    query = data
    metric_type = MetricType.L2
    ground_truth = utils.generate_groundtruth(query, data, metric_type)
    assert ground_truth.shape == (1000, 100)
    assert ground_truth.dtype == np.int64
    assert np.all(ground_truth >= 0)
    assert np.all(ground_truth < 1000)
    assert np.equal(ground_truth[:,0], np.arange(1000)).all()

def test_generate_groundtruth_without_faiss():
    # mock for faiss to remove knn_gpu function
    with mock.patch.dict('sys.modules', {'faiss': None}):
        test_generate_groundtruth()
