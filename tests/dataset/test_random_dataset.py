from os import path
from annb.dataset import RandomDataset
from annb import MetricType

def test_random_dataset_generic(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir('cache')
        dataset = RandomDataset('cache/random_dataset.h5', dimension=4, count=2000, normalize=True)
        assert dataset.dimension == 4
        assert dataset.count == 2000
        assert dataset.metric_type == MetricType.L2
        assert dataset.normalize == True
        assert path.exists('cache/random_dataset.h5')
        assert len(dataset.data) == 2000
        assert len(dataset.ground_truth) == 2000
        assert len(dataset.query_data) == 2000
