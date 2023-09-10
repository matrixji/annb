from .indexes import MetricType
from .dataset import AnnbHdf5Dataset, Hdf5Dataset, RandomDataset, BaseDataset

__version__ = "0.1.18"

# metric types
METRIC_TYPE_INNER_PRODUCT = MetricType.INNER_PRODUCT
METRIC_TYPE_L2 = MetricType.L2

__all__ = [
    "METRIC_TYPE_INNER_PRODUCT",
    "METRIC_TYPE_L2",
    "BaseDataset",
    "AnnbHdf5Dataset",
    "Hdf5Dataset",
    "RandomDataset",
]
