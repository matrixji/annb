from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List

import numpy as np


class MetricType(Enum):
    INNER_PRODUCT = 1
    L2 = 2

    @classmethod
    def from_text(cls, text: str):
        if text.lower() == 'inner_product':
            return cls.INNER_PRODUCT
        elif text.lower() == 'ip':
            return cls.INNER_PRODUCT
        elif text.lower() == 'euclidean':
            return cls.EUCLIDEAN
        elif text.lower() == 'l2':
            return cls.L2
        else:
            raise ValueError('Unknown metric type: {}'.format(text))


class IndexUnderTest(ABC):
    """
    Abstract class for the index.
    """
    def __init__(self, index_name: str, dimension: int, metric_type: MetricType, **kwargs):
        """
        :param index_name: Name of the index.
        :param dimension: Dimension of the index.
        :param metric_type: Metric type of the index.
        """
        self.name = index_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.kwargs = kwargs

    def verify(self) -> bool:
        """
        Verify the index.
        :return: True if the index is valid, False otherwise.

        This method is used to verify that the index is valid.
        Index under test should be able to verify itself with small data set.
        """
        return True

    def cleanup(self) -> None:
        """
        Cleanup the index and related data.
        """
        pass

    @abstractmethod
    def train(self, data: np.ndarray) -> None:
        """
        Train the index
        :param data: List of data to train the index with.
        """
        pass

    def warmup(self) -> None:
        """
        Warmup the index, called before search.
        """
        pass

    @abstractmethod
    def add(self, data: np.ndarray) -> None:
        """
        Add data to the index.
        :param data: List of data to add to the index.
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        """
        Search the index.
        :param query: Query data.
        :param k: Number of nearest neighbors to return.
        :return: List of nearest neighbors, distances and ids
        """
        pass


class IndexUnderTestFactory(ABC):
    """
    Abstract class for the index factory.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def setup(self) -> None:
        """
        Setup the index factory.
        """
        pass

    @abstractmethod
    def create(self, index_name: str, dimension: int, metric_type: MetricType, **kwargs) -> IndexUnderTest:
        """
        Create the index.
        :return: IndexUnderTest object.
        """
        pass
