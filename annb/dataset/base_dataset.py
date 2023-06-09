from abc import ABC, abstractmethod
import numpy as np

from ..anns.indexes import MetricType

class BaseDataset(ABC):

    metric_type = MetricType.L2
    dimension = 0
    count = 0
    normalize = True

    def __init__(self, cache, **kwargs):
        """
        :param dataset_name: Name of the dataset.
        """
        self.cache = cache
        self.kwargs = kwargs

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        Return dataset data.
        """
        pass

    @property
    @abstractmethod
    def ground_truth(self) -> np.ndarray:
        """
        Return dataset ground truth.
        """
        pass

    @property
    @abstractmethod
    def query_data(self) -> np.ndarray:
        """
        Return dataset query.
        """
        pass
