from os import path
import numpy as np
from annb.dataset import AnnbHdf5Dataset
from annb import MetricType

def test_hdf5_dataset_create(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir('cache')
        dims = 128
        count = 1000
        metric = MetricType.L2
        data = np.random.rand(count, dims).astype('float32')
        AnnbHdf5Dataset.create(path.join('cache', 'test.hd5'), metric, data)

        dataset = AnnbHdf5Dataset(path.join('cache', 'test.hd5'))
        assert dataset.dimension == dims
        assert dataset.count == count
        assert dataset.metric_type == metric
        assert dataset.normalized == False
        assert np.array_equal(dataset.data, data)
        assert np.array_equal(dataset.train, data)
        print(dataset.hd5_file.filename)
        assert dataset.hd5_file.filename == path.join('cache', 'test.hd5')