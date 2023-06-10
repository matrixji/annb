import numpy as np

from .utils import generate_groundtruth as utils_generate_groundtruth
from .hd5_dataset import Hd5Dataset
from ..anns.indexes import MetricType

class RandomDataset(Hd5Dataset):
    def __init__(self, file: str, **kwargs):
        try:
            hdfile = Hd5Dataset.load_hdf5(file)
        except (FileNotFoundError, RuntimeError):
            metric_type = MetricType.from_text(kwargs.get('metric', 'l2'))
            dims = int(kwargs.get('dimension', 256))
            count = int(kwargs.get('count', 1000000))
            normalize = kwargs.get('normalize', False)
            data = self.generate_data(dims, count, normalize)
            ground_truth = self.generate_groundtruth(data, data, metric_type)
            Hd5Dataset.create(file, metric_type, data, ground_truth, normalize=normalize)
            hdfile = file
        super().__init__(hdfile, **kwargs)

    @classmethod
    def generate_data(cls, dimension, count, normalize) -> np.ndarray:
        """
        Generate random data.
        """
        data = np.random.random((count, dimension)).astype(np.float32)
        if normalize:
            data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
        return data

    @classmethod
    def generate_groundtruth(cls, query, data, metric_type) -> np.ndarray:
        """
        Generate ground truth for dataset.
        """
        # return utils_generate_groundtruth(query, data, metric_type)
        return np.random.randint(0, data.shape[0], (query.shape[0], 100), dtype=np.int64)
