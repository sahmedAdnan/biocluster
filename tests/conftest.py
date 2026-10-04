"""Shared fixtures.

Every test here runs offline and on synthetic data: no model weights are
downloaded, no dataset is read. That keeps CI honest and fast, and means a
contributor can verify the pipeline wiring without the 7 GB of checkpoints the
real encoders need.
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def synthetic(rng):
    """Well-separated clusters in a 64-d unit-norm space, the shape of a real
    embedding matrix, small enough to cluster instantly."""
    n_classes, per_class, dim = 8, 25, 64
    centres = rng.normal(size=(n_classes, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    feats, labels = [], []
    for c in range(n_classes):
        pts = centres[c] + 0.05 * rng.normal(size=(per_class, dim))
        feats.append(pts)
        labels += [c] * per_class
    X = np.vstack(feats)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float64), np.array(labels), n_classes
