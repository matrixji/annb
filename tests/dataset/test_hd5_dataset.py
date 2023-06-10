from os import path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from annb.dataset import Hd5Dataset
from annb import MetricType

def test_hd5_dataset_create(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir('cache')
        dims = 128
        count = 1000
        metric = MetricType.L2
        data = np.random.rand(count, dims).astype('float32')
        nbrs = NearestNeighbors(n_neighbors=100, metric='euclidean').fit(data)
        ground_truth = nbrs.kneighbors(data, return_distance=False)[0]
        Hd5Dataset.create(path.join('cache', 'test.hd5'), metric, data, ground_truth)

        dataset = Hd5Dataset(path.join('cache', 'test.hd5'))
        assert dataset.dimension == dims
        assert dataset.count == count
        assert dataset.metric_type == metric
        assert dataset.normalize == False
        assert np.array_equal(dataset.data, data)
        assert np.array_equal(dataset.ground_truth, ground_truth)
        print(dataset.hd5_file.filename)
        assert dataset.hd5_file.filename == path.join('cache', 'test.hd5')