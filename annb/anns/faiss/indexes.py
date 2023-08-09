from typing import List, Tuple, Union
import numpy as np
import faiss
from annb.indexes import IndexUnderTest, IndexUnderTestFactory, MetricType


class FaissIndexUnderTest(IndexUnderTest):
    def __init__(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ):
        super().__init__(index_name, dimension, metric_type, **kwargs)
        self.index = self.create_index()

    def create_index(self) -> Union[faiss.Index, None]:
        faiss_metric = (
            faiss.METRIC_L2
            if self.metric_type == MetricType.L2
            else faiss.METRIC_INNER_PRODUCT
        )
        using_gpu = str(self.kwargs.get("gpu", "no")).lower() in [
            "yes",
            "true",
            "1",
            "on",
        ]
        index_string = self.kwargs.get("index", "flat")
        index = None
        if index_string == "flat":
            index = faiss.IndexFlat(self.dimension, faiss_metric)
        elif index_string == "ivfflat":
            quantizer = faiss.IndexFlat(self.dimension, faiss_metric)
            nlist = self.kwargs.get("nlist", 128)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss_metric)
            self.log.info("create index ivfflat with nlist: %s", nlist)
        elif index_string == "ivfpq":
            quantizer = faiss.IndexFlat(self.dimension, faiss_metric)
            nlist = self.kwargs.get("nlist", 128)
            m = self.kwargs.get("m", 8)
            nbits = self.kwargs.get("nbits", 8)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
            self.log.info("create index ivfpq with nlist: %s, m: %s, nbits: %s", nlist, m, nbits)
        elif index_string == "ivfsq":
            quantizer = faiss.IndexFlat(self.dimension, faiss_metric)
            nlist = self.kwargs.get("nlist", 128)
            index = faiss.IndexIVFScalarQuantizer(
                quantizer,
                self.dimension,
                nlist,
                faiss.ScalarQuantizer.QT_8bit,
            )
            self.log.info("create index ivfsq(Q8) with nlist: %s", nlist)
        else:
            index = faiss.index_factory(self.dimension, index_string, faiss_metric)

        self.log.debug(
            "use cpu index, use gpu: %s, faiss has gpu support: %s",
            using_gpu,
            self.support_gpu(),
        )
        if using_gpu and self.support_gpu():
            self.log.debug("copy index to gpu")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    @classmethod
    def support_gpu(cls) -> bool:
        return hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0

    def train(self, data: np.ndarray) -> None:
        return self.index.train(data)

    def add(self, data: np.ndarray) -> None:
        return self.index.add(data)

    def warmup(self) -> None:
        for _ in range(3):
            random_data = np.random.rand(10, self.dimension).astype("float32")
            random_data /= np.linalg.norm(random_data, axis=1)[:, None]
            self.search(random_data, 10)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        return self.index.search(query, k)

    def update_search_args(self, **kwargs):
        if "nprobe" in kwargs:
            self.index.nprobe = int(kwargs["nprobe"])

    def cleanup(self) -> None:
        self.index.reset()


class FaissIndexUnderTestFactory(IndexUnderTestFactory):
    def create(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ) -> FaissIndexUnderTest:
        return FaissIndexUnderTest(index_name, dimension, metric_type, **kwargs)


index_under_test_factory = FaissIndexUnderTestFactory
