"""The annotation stage: naming, the four uncertainty signals, and tiers.

Text embeddings are synthesised rather than encoded, so this runs without
downloading an encoder.
"""
import numpy as np
import pytest

from biocluster.annotation.annotator import (
    classify_confidence,
    compute_combined_uncertainty,
    run_annotation,
)
from biocluster.annotation.taxonomy import parse_taxonomy_from_names


@pytest.fixture
def annotation_inputs(synthetic):
    X, y, k = synthetic
    species = [f"Genus_{i} species_{i}" for i in range(k)]
    # One text vector per species, placed on that class's own centroid, so the
    # correct name is the nearest one.
    text = np.vstack([X[y == c].mean(axis=0) for c in range(k)])
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return X, y, species, text, parse_taxonomy_from_names(species)


def test_tier_thresholds_are_taken_from_config():
    u = np.array([0.0, 0.54, 0.55, 0.69, 0.70, 1.0])
    cfg = {"annotation": {"uncertainty": {"review_threshold": 0.55,
                                          "human_threshold": 0.70}}}
    assert classify_confidence(u, cfg) == [
        "auto_accept", "auto_accept", "review", "review",
        "human_required", "human_required"]


def test_combined_uncertainty_is_a_weighted_mean():
    one = np.ones(3)
    assert np.allclose(compute_combined_uncertainty(one, one, one, one), one)
    zero = np.zeros(3)
    assert np.allclose(compute_combined_uncertainty(zero, zero, zero, zero), zero)


def test_run_annotation_names_clusters_correctly(annotation_inputs):
    X, y, species, text, taxonomy = annotation_inputs
    cfg = {"annotation": {"uncertainty": {"review_threshold": 0.55,
                                          "human_threshold": 0.70}}}
    df = run_annotation(X, y.copy(), text, species, taxonomy, cfg)

    assert len(df) == len(X)
    for col in ("ai_annotation", "combined_uncertainty", "confidence_tier",
                "cluster_id"):
        assert col in df.columns
    # every cluster is a pure class here, so naming must be exact
    truth = [species[int(c)] for c in y]
    assert (df["ai_annotation"] == truth).all()
    assert set(df["confidence_tier"]) <= {"auto_accept", "review", "human_required"}


def test_noise_points_are_named_unknown_and_flagged(annotation_inputs):
    X, y, species, text, taxonomy = annotation_inputs
    ids = y.copy()
    ids[:10] = -1                                  # mark some points as noise
    cfg = {"annotation": {"uncertainty": {"review_threshold": 0.55,
                                          "human_threshold": 0.70}}}
    df = run_annotation(X, ids, text, species, taxonomy, cfg)

    noise = df[df["cluster_id"] == -1]
    assert len(noise) == 10
    assert (noise["ai_annotation"] == "Noise / Unknown").all()
    # noise carries maximal spatial/agreement/taxonomic uncertainty, so it can
    # never land in auto_accept
    assert "auto_accept" not in set(noise["confidence_tier"])
