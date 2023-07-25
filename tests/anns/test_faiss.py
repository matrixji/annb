import os
import subprocess

import numpy as np
from annb.anns.faiss.indexes import index_under_test_factory
from annb.anns.faiss.deploy import index_under_test_deployment
from annb.indexes import MetricType


def test_faiss_index_under_test():
    factory = index_under_test_factory()
    index_under_test = factory.create('IndexFlat', 4, MetricType.L2, index='flat')
    x = np.random.rand(100, 4).astype(np.float32)
    index_under_test.add(x)
    distances, ids = index_under_test.search(x[:3], 3)
    assert len(distances) == 3
    assert len(ids) == 3
    # the 1st column
    assert list(ids[:, 0]) == [0, 1, 2]


def test_faiss_index_under_test_factory():
    factory = index_under_test_factory()
    index_under_test = factory.create('IndexFlat', 4, MetricType.L2, index='flat')
    x = np.random.rand(100, 4).astype(np.float32)
    index_under_test.add(x)
    assert index_under_test.index.ntotal == 100


def test_faiss_index_deploy():
    deployment = index_under_test_deployment()
    deploy_type, ref = deployment.deploy()
    assert deploy_type == 'builtin'
    assert ref == ''


def test_faiss_index_deploy_with_pip(tmpdir):
    with tmpdir.as_cwd():
        deployment = index_under_test_deployment()
        deploy_type, venv_path = deployment.deploy(deployment_type='venv')
        assert deploy_type == 'venv'
        ret = subprocess.check_call([os.path.join(venv_path, 'bin', 'python'), '-c', 'import faiss'])
        assert ret == 0
