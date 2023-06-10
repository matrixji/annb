from os import path
from typing import Union
import h5py as h5
import numpy as np


from .base_dataset import BaseDataset
from ..anns.indexes import MetricType


class Hd5Dataset(BaseDataset):
    def __init__(self, file: Union[str, h5.File], **kwargs):
        """
        :param file: Path or hdf5 file to the dataset file.
        """
        super().__init__(**kwargs)
        if isinstance(file, str):
            self.hd5_file = self.load_hdf5(file)
        elif isinstance(file, h5.File):
            self.hd5_file = file
        self.metric_type = self.get_metric_type(self.hd5_file)
        self.dimension = self.get_dimension(self.hd5_file)
        self.count = self.get_count(self.hd5_file)
        self.normalize = self.get_normalize(self.hd5_file)
        self.dataset = {}

    @classmethod
    def load_hdf5(cls, cache_file: str) -> h5.File:
        """
        Load dataset from cache file.
        """
        if not cache_file or not path.exists(cache_file):
            raise FileNotFoundError(f'file {cache_file} not found.')
        hd = h5.File(cache_file, 'r')
        if not cls.get_done(hd):
            raise RuntimeError(f'file {cache_file} not done.')
        return hd

    @classmethod
    def create(cls, cache_file: str, metric_type: MetricType, data: np.ndarray, ground_truth: np.ndarray,
               query_data: Union[np.ndarray, None] = None, training_data: Union[np.ndarray, None] = None, normalize: bool = False):
        """
        Create dataset and save to cache file.

        :param cache_file: cache file path.
        :param metric_type: metric type.
        :param data: dataset data.
        :param ground_truth: dataset ground truth.
        :param query_data: dataset query data.
        :param training_data: dataset training data.
        :param normalize: normalize dataset data.
        """
        hd = h5.File(cache_file, 'w')
        hd.attrs['dimension'] = data.shape[1]
        hd.attrs['count'] = data.shape[0]
        hd.attrs['metric'] = str(metric_type.name).lower()
        hd.attrs['normalize'] = normalize
        hd.create_dataset('data', data=data)
        hd.create_dataset('ground_truth', data=ground_truth)
        if query_data is not None:
            hd.create_dataset('query_data', data=query_data)
        if training_data is not None:
            hd.create_dataset('training_data', data=training_data)
        hd.attrs['done'] = True
        hd.close()


    @classmethod
    def get_done(cls, hd: h5.File) -> bool:
        """
        Get dataset done.
        """
        if 'done' in hd.attrs:
            return hd.attrs['done']
        return False

    @classmethod
    def get_dimension(cls, hd: h5.File) -> int:
        """
        Get dataset dimension.
        """
        if 'dimension' in hd.attrs:
            return hd.attrs['dimension']
        if 'data' in hd:
            return len(np.ndarray(hd['data'])[0])
        return -1

    @classmethod
    def get_count(cls, hd: h5.File) -> int:
        """
        Get dataset count.
        """
        if 'count' in hd.attrs:
            return hd.attrs['count']
        if 'data' in hd:
            return len(np.ndarray(hd['data']))
        return -1

    @classmethod
    def get_metric_type(cls, hd: h5.File) -> MetricType:
        """
        Get dataset metric type.
        """
        if 'metric' in hd.attrs:
            return MetricType.from_text(str(hd.attrs['metric']))
        return MetricType.L2

    @classmethod
    def get_normalize(cls, hd: h5.File) -> bool:
        """
        Get dataset normalize.
        """
        if 'normalize' in hd.attrs:
            return hd.attrs['normalize']
        return False

    @property
    def data(self) -> np.ndarray:
        """
        Return dataset data.
        """
        if 'data' not in self.dataset:
            data = np.array(self.hd5_file['data'])
            self.dataset['data'] = data
        return self.dataset['data']

    @property
    def training_data(self) -> np.ndarray:
        """
        Return dataset training data.
        """
        if 'training_data' not in self.dataset:
            if 'training_data' in self.hd5_file:
                data = np.array(self.hd5_file['training_data'])
            else:
                data = np.array(self.hd5_file['data'])
            self.dataset['training_data'] = data
        return self.dataset['training_data']

    @property
    def ground_truth(self) -> np.ndarray:
        """
        Return dataset ground truth.
        """
        if 'ground_truth' not in self.dataset:
            data = np.array(self.hd5_file['ground_truth'])
            self.dataset['ground_truth'] = data
        return self.dataset['ground_truth']

    @property
    def query_data(self) -> np.ndarray:
        """
        Return dataset query.
        """
        if 'query_data' not in self.dataset:
            if 'query_data' in self.hd5_file:
                data = np.array(self.hd5_file['query_data'])
            else:
                nq = self.ground_truth.shape[0]
                data = np.array(self.hd5_file['data'][:nq])
            self.dataset['query_data'] = data
        return self.dataset['query_data']
