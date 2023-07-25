import numpy as np

from .hdf5_dataset import AnnbHdf5Dataset
from ..indexes import MetricType


class RandomDataset(AnnbHdf5Dataset):
    def __init__(self, file: str, **kwargs):
        metric_type = MetricType.from_text(kwargs.get('metric', 'l2'))
        dims = int(kwargs.get('dimension', 256))
        count = int(kwargs.get('count', 1000000))
        try:
            hdfile = AnnbHdf5Dataset.load_hdf5(file)
            temp_dataset = AnnbHdf5Dataset(hdfile)
            if temp_dataset.metric_type != metric_type:
                hdfile.close()
                raise RuntimeError(
                    f'Metric type mismatch: {temp_dataset.metric_type} != {metric_type}'
                )
            if temp_dataset.dimension != dims:
                hdfile.close()
                raise RuntimeError(
                    f'Dimension mismatch: {temp_dataset.dimension} != {dims}'
                )
            if temp_dataset.count != count:
                hdfile.close()
                raise RuntimeError(f'Count mismatch: {temp_dataset.count} != {count}')
        except (FileNotFoundError, RuntimeError):
            metric_type = MetricType.from_text(kwargs.get('metric', 'l2'))
            dims = int(kwargs.get('dimension', 256))
            count = int(kwargs.get('count', 1000000))
            normalize = kwargs.get('normalize', False)
            data = self.generate_data(dims, count, normalize)
            ground_truth = kwargs.get('ground_truth', True)
            AnnbHdf5Dataset.create(
                file, metric_type, data, normalize=normalize, ground_truth=ground_truth
            )
            hdfile = AnnbHdf5Dataset.load_hdf5(file)
        super().__init__(hdfile, **kwargs)

    @classmethod
    def generate_data(cls, dimension, count, normalize) -> np.ndarray:
        """
        Generate random data.
        """
        data = np.random.random((count, dimension)).astype(np.float32)
        return data
