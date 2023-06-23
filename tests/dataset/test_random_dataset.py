from os import path
from annb.dataset import RandomDataset
from annb import MetricType

def test_random_dataset_generic(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir('cache')
        dataset = RandomDataset('cache/random_dataset.h5', metric='ip', dimension=4, count=2000, normalize=True)
        assert dataset.dimension == 4
        assert dataset.count == 2000
        assert dataset.metric_type == MetricType.INNER_PRODUCT
        assert dataset.normalized == True
        assert path.exists('cache/random_dataset.h5')
        assert len(dataset.data) == 2000
        assert len(dataset.ground_truth_neighbors) == 2000
        assert len(dataset.ground_truth_distances) == 2000
        assert len(dataset.train) == 2000
