"""Clustering recovers synthetic structure, and the metrics agree."""
import numpy as np
import pytest

from biocluster.clustering.methods import run_clustering
from biocluster.evaluation.metrics import hungarian_accuracy


def _score(labels_true, ids):
    from sklearn.metrics import adjusted_rand_score
    mask = ids >= 0
    return adjusted_rand_score(labels_true[mask], ids[mask])


@pytest.mark.parametrize("method,params", [
    ("kmeans",        {"n_clusters": 8, "n_init": 10}),
    ("agglomerative", {"n_clusters": 8, "linkage": "ward"}),
    ("gmm",           {"n_clusters": 8, "covariance_type": "diag"}),
    ("spectral",      {"n_clusters": 8, "n_neighbors": 10}),
])
def test_partition_methods_recover_clusters(synthetic, method, params):
    X, y, k = synthetic
    ids = np.asarray(run_clustering(method, X, params))
    assert len(ids) == len(X)
    assert _score(y, ids) > 0.8


def test_hdbscan_recovers_clusters(synthetic):
    pytest.importorskip("hdbscan")
    X, y, _ = synthetic
    ids = np.asarray(run_clustering("hdbscan", X,
                                    {"min_cluster_size": 5, "min_samples": 3,
                                     "cluster_selection_method": "eom"}))
    assert _score(y, ids) > 0.8


def test_dbscan_noise_is_negative_one(synthetic):
    """eps is an absolute radius, so it must come from the data rather than a
    guessed constant: the median 4th-neighbour distance sits inside the clusters
    but below the gaps between them."""
    from sklearn.neighbors import NearestNeighbors

    X, y, _ = synthetic
    d, _ = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)
    eps = float(np.median(d[:, 4]) * 1.5)

    ids = np.asarray(run_clustering("dbscan", X, {"eps": eps, "min_samples": 4}))
    assert len(ids) == len(X)
    assert ids.min() >= -1, "noise must be -1; no other negative id may appear"
    assert len(set(np.unique(ids)) - {-1}) >= 2, "expected real clusters"
    assert _score(y, ids) > 0.8


def test_hungarian_accuracy_is_perfect_on_a_perfect_partition():
    y = np.array([0, 0, 1, 1, 2, 2])
    acc, mapping = hungarian_accuracy(y, np.array([2, 2, 0, 0, 1, 1]))
    assert acc == 1.0
    assert len(mapping) >= 3


def test_unknown_method_raises(synthetic):
    X, _, _ = synthetic
    with pytest.raises(ValueError):
        run_clustering("not_a_method", X, {})
