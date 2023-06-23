from time import monotonic
from typing import Tuple
import numpy as np
from ..anns.indexes import MetricType


def duration_execution(stage, func, *args, **kwargs):
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
        from faiss import knn_gpu, METRIC_L2, METRIC_INNER_PRODUCT, StandardGpuResources
        metric = METRIC_L2 if metric_type == MetricType.L2 else METRIC_INNER_PRODUCT
        res = StandardGpuResources()
        return duration_execution('generate_groundtruth/knn_gpu', knn_gpu, res, query[:query_max], data, k, metric=metric)
    except ImportError:
        pass

    try:
        from faiss import knn, METRIC_L2, METRIC_INNER_PRODUCT
        metric = METRIC_L2 if metric_type == MetricType.L2 else METRIC_INNER_PRODUCT
        return duration_execution('generate_groundtruth/knn', knn, query[:query_max], data, k, metric=metric)
    except ImportError:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors
        metric_text = 'l2' if metric_type == MetricType.L2 else 'cosine'
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric_text, n_jobs=-1).fit(data)
        return duration_execution('generate_groundtruth/sklearn', nbrs.kneighbors, query[:query_max])
    except ImportError:
        pass

    raise RuntimeError('No knn implementation found')
