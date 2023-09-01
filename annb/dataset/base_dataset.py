from abc import ABC, abstractmethod
import numpy as np

from ..indexes import MetricType


class BaseDataset(ABC):
    name = 'Dataset'
    metric_type = MetricType.L2
    dimension = 0
    count = 0

    def __init__(self, **kwargs):
        """
        :param dataset_name: Name of the dataset.
        """
        self.kwargs = kwargs

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        Return the dataset to be added to the index.
        """
        pass

    @property
    @abstractmethod
    def train(self) -> np.ndarray:
        """
        Return dataset to be used for training.
        could use different dataset for training and adding.
        """
        pass

    @property
    @abstractmethod
    def test(self) -> np.ndarray:
        """
        Return dataset to be used for search testing.
        """
        pass

    @property
    @abstractmethod
    def ground_truth_neighbors(self) -> np.ndarray:
        """
        Return ground truth for dataset. ID labels of nearest neighbors.
        """
        pass

    @property
    @abstractmethod
    def ground_truth_distances(self) -> np.ndarray:
        """
        Return ground truth for dataset. Distances to nearest neighbors.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Fit the dataset for training/add/query.
        """
        pass