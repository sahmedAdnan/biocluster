"""Reduction modes behave as documented on synthetic data."""
import numpy as np
import pytest

from biocluster.reduction.reduction import reduce_features


def test_none_is_identity(synthetic):
    X, _, _ = synthetic
    out, reducer = reduce_features(X, {"reduction": {"method": "none"}})
    assert out.shape == X.shape
    assert reducer is None
    assert np.allclose(out, X)


def test_pca_shape_and_variance(synthetic):
    X, _, _ = synthetic
    out, pca = reduce_features(
        X, {"reduction": {"method": "pca", "n_components": 10}}, random_state=42)
    assert out.shape == (X.shape[0], 10)
    assert pca.explained_variance_ratio_.sum() > 0.5


def test_pca_is_deterministic(synthetic):
    X, _, _ = synthetic
    cfg = {"reduction": {"method": "pca", "n_components": 5}}
    a, _ = reduce_features(X, cfg, random_state=42)
    b, _ = reduce_features(X, cfg, random_state=42)
    assert np.allclose(a, b)


def test_umap_shape(synthetic):
    pytest.importorskip("umap")
    X, _, _ = synthetic
    out, _ = reduce_features(
        X, {"reduction": {"method": "umap", "n_components": 5, "n_neighbors": 10,
                          "min_dist": 0.1, "metric": "cosine"}}, random_state=42)
    assert out.shape == (X.shape[0], 5)


def test_unknown_mode_raises(synthetic):
    X, _, _ = synthetic
    with pytest.raises(ValueError):
        reduce_features(X, {"reduction": {"method": "nope"}})
