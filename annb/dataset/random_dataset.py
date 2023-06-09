from os import path
from typing import Union
import h5py as h5
import faiss
import numpy as np


from .base_dataset import BaseDataset
from ..anns.indexes import MetricType

class RandomDataset(BaseDataset):

    def __init__(self, cache, **kwargs):
        super().__init__(cache, **kwargs)
        self.metric_type = MetricType.from_text(kwargs.get('metric', 'l2'))
        self.dimension = int(kwargs.get('dimension', 128))
        self.count = int(kwargs.get('count', 1000000))
        self.normalize = kwargs.get('normalize', True)
        self.hd5_file = self.load_from_cache(cache)
        if not self.hd5_file:
            self.hd5_file = self.generate(cache)

    def load_from_cache(self, cache_file: str) -> Union[h5.File, None]:
        """
        Load dataset from cache file.
        """
        if not path.exists(cache_file):
            return None
        hd = h5.File(cache_file, 'r')
        dimension = hd.attrs['dimension'] if 'dimension' in hd.attrs else len(np.ndarray(hd['data'])[0])
        count = hd.attrs['count'] if 'count' in hd.attrs else len(np.ndarray(hd['data']))
        metric = MetricType.from_text(str(hd.attrs['metric'])) if 'metric' in hd.attrs else MetricType.L2
        normalize = hd.attrs['normalize'] if 'normalize' in hd.attrs else True
        if dimension != self.dimension or count != self.count or metric != self.metric_type or normalize != self.normalize:
            hd.close()
            return None
        if 'done' not in hd.attrs or not hd.attrs['done']:
            hd.close()
            return None
        return hd

    def generate(self, cache_file: str) -> Union[h5.File, None]:
        """
        Generate dataset and save to cache file.
        """
        hd = h5.File(cache_file, 'w')
        hd.attrs['dimension'] = self.dimension
        hd.attrs['count'] = self.count
        hd.attrs['metric'] = str(self.metric_type.name).lower()
        hd.attrs['normalize'] = self.normalize
        data = np.random.rand(self.count, self.dimension).astype('float32')
        if self.normalize:
            data = data / np.linalg.norm(data, axis=1)[:, None]
        hd.create_dataset('data', data=data)
        ground_truth = self.generate_groundtruth(data)
        hd.create_dataset('ground_truth', data=ground_truth)
        hd.attrs['done'] = True
        hd.close()
        return self.load_from_cache(cache_file)
    
    def generate_groundtruth(self, data):
        """
        Generate ground truth for dataset.
        """
        knn_func = faiss.knn
        metric = faiss.METRIC_L2 if self.metric_type == MetricType.L2 else faiss.METRIC_INNER_PRODUCT
        nq = min(16384, self.count)
        if hasattr(faiss, 'knn_gpu') and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            knn_func = lambda x, y, k, m: faiss.knn_gpu(res, x, y, k, metric=m)
        return knn_func(data[:nq], data, 100, metric)[1]

    @property
    def data(self) -> np.ndarray:
        """
        Return dataset data.
        """
        return np.array(self.hd5_file['data']).reshape(-1, self.dimension)
    
    @property
    def ground_truth(self) -> np.ndarray:
        """
        Return dataset ground truth.
        """
        return np.array(self.hd5_file['ground_truth']).reshape(-1, 100)
    
    @property
    def query_data(self) -> np.ndarray:
        """
        Return dataset query.
        """
        if 'query' in self.hd5_file:
            return np.array(self.hd5_file['query']).reshape(-1, self.dimension)
        return np.array(self.hd5_file['data'][:self.ground_truth.shape[0]]).reshape(-1, self.dimension)
