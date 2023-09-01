from os import path
from typing import Union
import h5py as h5
import numpy as np

from ..indexes import MetricType
from .base_dataset import BaseDataset
from .utils import generate_groundtruth


class Hdf5Dataset(BaseDataset):
    """
    Dataset from hdf5 file.
    This dataset is for load dataset from hdf5 file from ann-benchmarks.
    """

    def __init__(self, file: Union[str, h5.File], **kwargs):
        """
        :param file: Path or hdf5 file to the dataset file.
        """
        super().__init__(**kwargs)
        if isinstance(file, str):
            self.hd5_file = self.load_hdf5(file)
        elif isinstance(file, h5.File):
            self.hd5_file = file
        self.validate()
        # load the data
        self.data_ = np.array(self.hd5_file['train'])
        self.test_data_ = np.array(self.hd5_file['test'])
        if 'dimension' in self.hd5_file.attrs:
            self.dimension = self.hd5_file.attrs['dimension']
            self.count = self.data.shape[0] / self.dimension
        else:
            self.dimension = self.data.shape[1]
            self.count = self.data.shape[0]
        self.metric_type = MetricType.from_text(str(self.hd5_file.attrs['distance']))
        self.name = path.basename(self.hd5_file.filename)


    def __str__(self) -> str:
        return (
            f'<{self.__class__.__name__}({self.name}, m={self.metric_type.name},'
            f' d={self.dimension}, nb={self.count})>'
        )

    def validate(self):
        """
        Validate dataset.
        """
        if 'distance' not in self.hd5_file.attrs:
            raise RuntimeError('dataset file not found distance attribute.')
        distance_text = str(self.hd5_file.attrs['distance'])
        self.metric_type = MetricType.from_text(distance_text)

        for required_dataset in ('distances', 'neighbors', 'test', 'train'):
            if required_dataset not in self.hd5_file:
                raise RuntimeError(
                    f'dataset file not found {required_dataset} dataset.'
                )

    @classmethod
    def load_hdf5(cls, cache_file: str) -> h5.File:
        """
        Load dataset from cache file.
        """
        if not cache_file or not path.exists(cache_file):
            raise FileNotFoundError(f'file {cache_file} not found.')
        hd = h5.File(cache_file)
        return hd

    @classmethod
    def create(
        cls,
        output: str,
        metric: MetricType,
        data_and_train: np.ndarray,
        test: Union[np.ndarray, None] = None,
        distances: Union[np.ndarray, None] = None,
        neighbors: Union[np.ndarray, None] = None,
        normalize: bool = False,
        ground_truth: bool = True,
        extra_attrs: dict = None,
    ):
        """
        Create dataset and save to file
        :param output: Output file path.
        :param metric: Metric type.
        :param data_and_train: Data and train data.
        :param test: Test data.
        :param distances: Distances data.
        :param neighbors: Neighbors data.
        :param normalize: Normalize data.
        :param ground_truth: Generate ground truth.
        :param extra_attrs: Extra attributes.
        """

        def get_distance_text(the_metric: MetricType) -> str:
            return 'euclidean' if the_metric == MetricType.L2 else 'angular'

        if normalize and metric == MetricType.L2:
            raise ValueError('normalize only support angular/ip metric.')
        if normalize and (distances is not None or neighbors is not None):
            raise ValueError(
                'normalize is set, but distances/neighbors data is not None.'
            )

        hd = h5.File(output, 'w')
        hd.attrs['distance'] = get_distance_text(metric)
        extra_attrs = extra_attrs or {}
        for k, v in extra_attrs.items():
            hd.attrs[k] = v

        # do normalize for data and train
        if normalize:
            data_and_train /= np.linalg.norm(data_and_train, axis=1)[:, np.newaxis]
        hd.create_dataset('train', data=data_and_train)

        if test is None:
            test_size = min(data_and_train.shape[0], 10000)
            test = data_and_train[:test_size]
            neighbors, distances = None, None
        else:
            # normalize for test data
            test /= np.linalg.norm(test, axis=1)[:, np.newaxis]

        if neighbors is None or distances is None:
            if ground_truth:
                distances, neighbors = generate_groundtruth(
                    test, data_and_train, metric
                )
            else:
                # return forge data, if no ground truth is needed
                neighbors = np.zeros((test.shape[0], 1), dtype=np.int64)
                distances = np.zeros((test.shape[0], 1), dtype=np.float32)

        hd.create_dataset('test', data=test)
        hd.create_dataset('neighbors', data=neighbors)
        hd.create_dataset('distances', data=distances)
        hd.close()

    @property
    def data(self) -> np.ndarray:
        # using train dataset as data
        return self.data_

    @property
    def train(self) -> np.ndarray:
        return self.data_

    @property
    def test(self) -> np.ndarray:
        """
        Return query test dataset.
        """
        return self.test_data_

    @property
    def ground_truth_distances(self) -> np.ndarray:
        """
        Return dataset data.
        """
        return np.array(self.hd5_file['distances'])

    @property
    def ground_truth_neighbors(self) -> np.ndarray:
        """
        Return dataset ground truth.
        """
        return np.array(self.hd5_file['neighbors'])


class AnnbHdf5Dataset(Hdf5Dataset):
    """
    Dataset from hdf5 file, with some extra attributes from annb.
    """

    def __init__(self, file: Union[str, h5.File], **kwargs):
        """
        :param file: Path or hdf5 file to the dataset file.
        """
        super().__init__(file, **kwargs)
        self.normalized = self.hd5_file.attrs.get('normalized', False)

    def __str__(self) -> str:
        return (
            f'<{self.__class__.__name__}({self.name},'
            f' m={self.metric_type.name}, d={self.dimension},'
            f' nb={self.count}, normalized={self.normalized})>'
        )

    @classmethod
    def create(
        cls,
        output: str,
        metric_type: MetricType,
        data_and_train: np.ndarray,
        test: Union[np.ndarray, None] = None,
        distances: Union[np.ndarray, None] = None,
        neighbors: Union[np.ndarray, None] = None,
        normalize: bool = False,
        ground_truth: bool = True,
        extra_attrs: dict = None,
    ):
        extra_attrs = extra_attrs or {}
        extra_attrs['normalized'] = normalize
        return Hdf5Dataset.create(
            output,
            metric_type,
            data_and_train,
            test,
            distances,
            neighbors,
            normalize,
            ground_truth,
            extra_attrs,
        )

    def fit(self):
        # format float32 for float index
        if self.metric_type in (MetricType.L2, MetricType.INNER_PRODUCT):
            if self.data_.dtype != np.float32:
                self.data_ = self.data_.astype(np.float32)
        # normalize for angular/ip metric
        if not self.normalized and self.metric_type == MetricType.INNER_PRODUCT:
            self.data_ /= np.linalg.norm(self.data_, axis=1)[:, np.newaxis]
            self.test_data_ /= np.linalg.norm(self.test_data_, axis=1)[:, np.newaxis]
            self.normalized = True
