from os import path
from typing import Union
import h5py as h5
import numpy as np


from ..anns.indexes import MetricType
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
        self.dimension = self.data.shape[1]
        self.count = self.data.shape[0]
        self.metric_type = MetricType.from_text(str(self.hd5_file.attrs["distance"]))
        self.name = path.basename(self.hd5_file.filename)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}({self.name}, m={self.metric_type.name}, d={self.dimension}, nb={self.count})>"

    def validate(self):
        """
        Validate dataset.
        """
        if "distance" not in self.hd5_file.attrs:
            raise RuntimeError("dataset file not found distance attribute.")
        distance_text = str(self.hd5_file.attrs["distance"])
        self.metric_type = MetricType.from_text(distance_text)

        for required_dataset in ("distances", "neighbors", "test", "train"):
            if required_dataset not in self.hd5_file:
                raise RuntimeError(
                    f"dataset file not found {required_dataset} dataset."
                )

    @classmethod
    def load_hdf5(cls, cache_file: str) -> h5.File:
        """
        Load dataset from cache file.
        """
        if not cache_file or not path.exists(cache_file):
            raise FileNotFoundError(f"file {cache_file} not found.")
        hd = h5.File(cache_file, "r")
        return hd

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
        extra_attrs: dict = {},
    ):
        """
        Create dataset and save to file

        :param output: Output file.
        :param metric_type: Metric type.
        :param distances: Distances.
        :param neighbors: Neighbors.
        :param test: Test data.
        :param train: Train data.
        :param normalize: Normalize data.
        :param extra_attrs: Extra attributes.
        """

        def get_distance_text(metric_type: MetricType) -> str:
            return "euclidean" if metric_type == MetricType.L2 else "angular"

        if normalize and metric_type == MetricType.L2:
            raise ValueError("normalize only support angular/ip metric.")
        if normalize and (distances is not None or neighbors is not None):
            raise ValueError(
                "normalize is set, but distances/neighbors data is not None."
            )

        hd = h5.File(output, "w")
        hd.attrs["distance"] = get_distance_text(metric_type)
        for k, v in extra_attrs.items():
            hd.attrs[k] = v

        # do normalize for data and train
        if normalize:
            data_and_train /= np.linalg.norm(data_and_train, axis=1)[:, np.newaxis]
        hd.create_dataset("train", data=data_and_train)

        if test is None:
            test_size = min(data_and_train.shape[0], 10000)
            test = data_and_train[:test_size]
            neighbors, distances = None, None
        else:
            # normalize for test data
            test /= np.linalg.norm(test, axis=1)[:, np.newaxis]

        if neighbors is None or distances is None:
            if ground_truth:
                neighbors, distances = generate_groundtruth(
                    test, data_and_train, metric_type
                )
            else:
                # return forge data, if no ground truth is needed
                neighbors = np.zeros((test.shape[0], 1), dtype=np.int64)
                distances = np.zeros((test.shape[0], 1), dtype=np.float32)

        hd.create_dataset("test", data=test)
        hd.create_dataset("neighbors", data=neighbors)
        hd.create_dataset("distances", data=distances)
        hd.close()

    @property
    def data(self) -> np.ndarray:
        # using train dataset as data
        return np.array(self.hd5_file["train"])

    @property
    def train(self) -> np.ndarray:
        return np.array(self.hd5_file["train"])

    @property
    def test(self) -> np.ndarray:
        """
        Return quert test dataset.
        """
        return np.array(self.hd5_file["test"])

    @property
    def ground_truth_distances(self) -> np.ndarray:
        """
        Return dataset data.
        """
        return np.array(self.hd5_file["distances"])

    @property
    def ground_truth_neighbors(self) -> np.ndarray:
        """
        Return dataset ground truth.
        """
        return np.array(self.hd5_file["neighbors"])


class AnnbHdf5Dataset(Hdf5Dataset):
    """
    Dataset from hdf5 file, with some extra attributes from annb.
    """

    def __init__(self, file: Union[str, h5.File], **kwargs):
        """
        :param file: Path or hdf5 file to the dataset file.
        """
        super().__init__(file, **kwargs)
        self.normalized = self.hd5_file.attrs.get("normalized", False)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}({self.name}, m={self.metric_type.name}, d={self.dimension}, nb={self.count}, normalized={self.normalized})>"

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
        extra_attrs: dict = {},
    ):
        extra_attrs["normalized"] = normalize
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
