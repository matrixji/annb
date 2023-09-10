from typing import List, Tuple, Union
import numpy as np
from pymilvus import Collection, connections, CollectionSchema, FieldSchema, DataType
from annb.indexes import IndexUnderTest, IndexUnderTestFactory, MetricType


class MilvusIndexUnderTest(IndexUnderTest):
    def __init__(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ):
        super().__init__(index_name, dimension, metric_type, **kwargs)
        self.connect()
        self.collection = self.create_collection()
        self.search_param = self.get_search_param()
        self.count = 0

    def get_search_param(self) -> dict:
        metric_type_text = self.get_index_param()['metric_type']
        search_param = {
            'metric_type': metric_type_text,
            'params': {}
        }
        if 'IVF' in metric_type_text:
            search_param['params']['nprobe'] = int(self.kwargs.get("nprobe", 128))
        if 'HNWS' in metric_type_text:
            search_param['params']['ef'] = int(self.kwargs.get("ef", 128))
        return search_param

    def get_index_param(self) -> str:
        index = self.kwargs.get("index", "IVF_FLAT")
        use_gpu = str(self.kwargs.get("gpu", "no")).lower() in [
            "yes",
            "true",
            "1",
            "on",
        ]
        use_gpu = use_gpu or 'GPU' in index
        metric_type_text = 'L2'
        if self.metric_type == MetricType.INNER_PRODUCT:
            metric_type_text = 'IP'
        #
        nlist = self.kwargs.get("nlist", 128)
        params = {}
        index_type = 'IVF_FLAT'
        if index in ('FLAT', 'flat'):
            index_type = 'FLAT'
        elif index in ('IVF_FLAT', 'ivfflat', 'GPU_IVF_FLAT'):
            if use_gpu:
                index_type = 'GPU_IVF_FLAT'
            else:
                index_type = 'IVF_FLAT'
            params = {
                'nlist': nlist,
            }
        elif index in ('IVF_SQ8', 'ivfsq8', 'ivfsq'):
            index_type = 'IVF_SQ8'
            params = {
                'nlist': nlist,
            }
        elif index in ('IVF_PQ', 'ivfpq', 'GPU_IVF_PQ'):
            if use_gpu:
                index_type = 'GPU_IVF_PQ'
            else:
                index_type = 'IVF_PQ'
            params = {
                'nlist': nlist,
                'm': int(self.kwargs.get("m", 16)),
                'nbits': int(self.kwargs.get("nbits", 8)),
            }
        elif index in ('HNSW', 'hnsw'):
            index_type = 'HNSW'
            params = {
                'M': int(self.kwargs.get("M", 16)),
                'efConstruction': int(self.kwargs.get("efConstruction", 256)),
            }
        return {
            'index_type': index_type,
            'metric_type': metric_type_text,
            'params': params,
        }

    def connect(self):
        uri = self.kwargs.get("uri", "http://localhost:19530")
        token = self.kwargs.get("token", "")
        connections.connect(uri=uri, token=token)

    def create_collection(self) -> Union[Collection, None]:
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name='id',
                    dtype=DataType.INT64,
                    is_primary=True,
                ),
                FieldSchema(
                    name='vector',
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                )
            ],
            description="annb benchmark collection",
        )
        try:
            Collection('annb_collection').drop()
        except:
            pass
        return Collection(
            name='annb_collection',
            schema=schema,
            using='default',
            consistent_level='Strong',
        )

    def train(self, data: np.ndarray) -> None:
        pass

    def add(self, data: np.ndarray) -> None:
        add_count = data.shape[0]
        step_size = 1000000 // self.dimension
        for i in range(0, add_count, step_size):
            step_data = data[i : i + step_size]
            step_count = step_data.shape[0]
            ids = list(range(self.count, self.count + step_count))
            self.collection.insert([ids, step_data.tolist()])
            self.count += step_count

    def warmup(self) -> None:
        index_params = self.get_index_param()
        self.collection.create_index(
            field_name='vector',
            index_params=index_params,
            index_name='annb_benchmark_index'
        )
        self.collection.load()
        for _ in range(3):
            random_data = np.random.rand(10, self.dimension).astype("float32")
            random_data /= np.linalg.norm(random_data, axis=1)[:, None]
            self.search(random_data, 10)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        result = self.collection.search(
            data=query,
            anns_field='vector',
            param=self.search_param,
            limit=k,
            consistency_level='Strong'
        )
        distances = []
        ids = []
        for r in result:
            distances.append(r.distances)
            ids.append(r.ids)
        return distances, ids

    def update_search_args(self, **kwargs):
        if "nprobe" in kwargs:
            self.search_param['params']['nprobe'] = kwargs['nprobe']
        if "ef" in kwargs:
            self.search_param['params']['ef'] = kwargs['ef']

    def cleanup(self) -> None:
        self.collection = self.create_collection()
        self.count = 0


class MilvusIndexUnderTestFactory(IndexUnderTestFactory):
    def create(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ) -> MilvusIndexUnderTest:
        return MilvusIndexUnderTest(index_name, dimension, metric_type, **kwargs)


index_under_test_factory = MilvusIndexUnderTestFactory
