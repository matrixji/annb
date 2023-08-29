import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List, Union, Dict
from hashlib import sha1
from logging import getLogger

import numpy as np
from .envs import get_run_dir


class MetricType(Enum):
    INNER_PRODUCT = 1
    L2 = 2
    JACCARD = 3

    @classmethod
    def from_text(cls, text: str):
        if text.lower() == 'inner_product':
            return cls.INNER_PRODUCT
        elif text.lower() == 'ip':
            return cls.INNER_PRODUCT
        elif text.lower() == 'angular':
            return cls.INNER_PRODUCT
        elif text.lower() == 'euclidean':
            return cls.L2
        elif text.lower() == 'l2':
            return cls.L2
        elif text.lower() == 'jaccard':
            return cls.JACCARD
        else:
            raise ValueError('Unknown metric type: {}'.format(text))


class IndexUnderTest(ABC):
    """
    Abstract class for the index.
    """

    def __init__(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ):
        """
        :param index_name: Name of the index.
        :param dimension: Dimension of the index.
        :param metric_type: Metric type of the index.
        """
        self.name = index_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.kwargs = kwargs
        self.log = getLogger('annb')

    def verify(self) -> bool:
        """
        Verify the index.
        :return: True if the index is valid, False otherwise.

        This method is used to verify that the index is valid.
        Index under test should be able to verify itself with small data set.
        """
        return self.metric_type in (MetricType.L2, MetricType.INNER_PRODUCT)

    @abstractmethod
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

    @abstractmethod
    def update_search_args(self, **kwargs) -> None:
        """
        Update search arguments.
        :param kwargs: Search arguments.
        """
        pass


class IndexUnderTestFactory(ABC):
    """
    Abstract class for the index factory.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def create(
        self, index_name: str, dimension: int, metric_type: MetricType, **kwargs
    ) -> IndexUnderTest:
        """
        Create the index.
        :return: IndexUnderTest object.
        """
        pass


class IndexUnderTestDeployment:
    """
    Deployment for index
    """

    def deploy(self, **kwargs) -> Tuple[str, str]:
        """
        Deploy the index.
        :return: Tuple of deployment type and reference.

        if deployment type is builtin, reference is empty string.
            return ('builtin', '')
        if deployment type is docker, reference is docker image name.
            return ('docker', 'docker_image_name')
        if deployment type is use conda, reference is conda environment path.
            return ('conda', 'conda_env_path')
        if deployment type is use venv, reference is venv environment path.
            return ('venv', 'venv_path')
        """
        deployment_type = kwargs.get('deployment_type', 'builtin')
        if deployment_type == 'docker':
            if 'image' not in kwargs:
                raise ValueError('Docker image is not specified for docker deployment')
            if 'dockerfile' in kwargs:
                self.build_docker_image(kwargs['dockerfile'], kwargs['image'])
            return 'docker', kwargs['image']
        if deployment_type == 'venv':
            if 'requirements' not in kwargs:
                raise ValueError(
                    'Requirements file is not specified for venv deployment'
                )
            venv_path = self.create_venv(kwargs['requirements'])
            return 'venv', venv_path
        if deployment_type == 'conda':
            if 'environments' not in kwargs:
                raise ValueError(
                    'Environments file is not specified for conda deployment'
                )
            conda_env_path = self.create_conda_env(kwargs['environments'])
            return 'conda', conda_env_path
        return 'builtin', ''

    @classmethod
    def build_docker_image(cls, dockerfile: str, image: str) -> None:
        """
        Build docker image.
        :param dockerfile: Dockerfile to build docker image.
        :param image: Docker image name.
        """
        import docker

        client = docker.from_env()
        client.images.build(path=dockerfile, tag=image)

    @classmethod
    def create_venv(cls, requirements: Union[List[str], str]) -> str:
        import venv
        import subprocess

        # read requirements if it is a file
        if isinstance(requirements, str):
            if os.path.isfile(requirements):
                with open(requirements, 'r') as f:
                    requirements = f.read()
            requirements = requirements.split('\n')
        requirements = sorted(requirements)
        sha1sum = sha1()
        for text in requirements:
            sha1sum.update(text.encode('utf-8'))
        venv_path = os.path.join(get_run_dir(), 'venv-{}'.format(sha1sum.hexdigest()))
        if not os.path.isfile(os.path.join(venv_path, 'bin', 'python')):
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(venv_path)
        requirements_path = os.path.join(venv_path, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        # install
        ret = subprocess.check_call(
            [os.path.join(venv_path, 'bin', 'pip'), 'install', '-r', requirements_path]
        )
        if ret != 0:
            raise RuntimeError('Failed to install requirements')
        return venv_path

    @classmethod
    def create_conda_env(cls, environments: Union[Dict, str]) -> str:
        import subprocess
        import yaml

        if isinstance(environments, str):
            if os.path.isfile(environments):
                with open(environments, 'r') as f:
                    environments = f.read()
            environments = yaml.loads(environments, Loader=yaml.FullLoader)
        conda_name = environments['name']
        conda_path = os.path.join(get_run_dir(), 'conda-{}'.format(conda_name))
        conda_yaml_path = conda_path + '.yaml'
        with open(conda_yaml_path, 'w') as f:
            yaml.dump(environments, f)
        # create conda
        cls.get_conda_executable()
        ret = subprocess.check_call(
            ['conda', 'env', 'create', '-f', conda_yaml_path, '-p', conda_path]
        )
        if ret != 0:
            raise RuntimeError('Failed to create conda env')
        return conda_path

    @classmethod
    def get_conda_executable(cls) -> str:
        home = os.environ.get('HOME', '')
        if home:
            for conda_path in [
                os.path.join(home, 'miniconda3', 'bin', 'conda'),
                os.path.join(home, 'anaconda3', 'bin', 'conda'),
                os.path.join(home, 'anaconda', 'bin', 'conda'),
            ]:
                if os.path.isfile(conda_path):
                    return conda_path
        return 'conda'
