from time import monotonic
from typing import Tuple
import numpy as np
from ..indexes import MetricType


def execution(stage, func, *args, **kwargs):
    """
    Measure execution time of the function.

    :param stage: Stage name
    :param func: Function to measure
    :param args: Function args
    :param kwargs: Function kwargs

    :return: Function result
    """
    # duration_stage = stage.split('#')[0]
    started = monotonic()
    res = func(*args, **kwargs)
    duration = monotonic() - started
    print('Stage: {}, duration: {:.3f}ms'.format(stage, duration * 1000.0))
    return res


def generate_groundtruth(query, data, metric_type) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground truth for dataset.

    :param query: Query data
    :param data: Dataset data
    :param metric_type: Metric type

    :return: Ground truth
    """
    query_max = min(16384, query.shape[0])
    k = 100
    try:
        from faiss import knn_gpu
    except ImportError:
        knn_gpu = None

    try:
        from faiss import StandardGpuResources
    except ImportError:
        StandardGpuResources = None  # noqa

    if knn_gpu and StandardGpuResources:
        from faiss import METRIC_L2, METRIC_INNER_PRODUCT

        metric = METRIC_L2 if metric_type == MetricType.L2 else METRIC_INNER_PRODUCT
        res = StandardGpuResources()
        return execution(
            'generate_groundtruth/knn_gpu',
            knn_gpu,
            res,
            query[:query_max],
            data,
            k,
            metric=metric,
        )
    try:
        from faiss import knn
    except ImportError:
        knn = None

    if knn:
        from faiss import METRIC_L2, METRIC_INNER_PRODUCT

        metric = METRIC_L2 if metric_type == MetricType.L2 else METRIC_INNER_PRODUCT
        return execution(
            'generate_groundtruth/knn', knn, query[:query_max], data, k, metric=metric
        )

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        NearestNeighbors = None  # noqa

    if NearestNeighbors:
        metric_text = 'l2' if metric_type == MetricType.L2 else 'cosine'
        nbs = NearestNeighbors(n_neighbors=k, metric=metric_text, n_jobs=-1).fit(data)
        return execution(
            'generate_groundtruth/sklearn', nbs.kneighbors, query[:query_max]
        )

    raise RuntimeError('No knn implementation found')
