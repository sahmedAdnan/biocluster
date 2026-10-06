"""The taxonomic signal reports whether it had any evidence.

A value of 0.0 is ambiguous: it means either "one family, genuinely pure" or
"no family could be resolved". The signal itself stays 0.0 in both cases so
the combined uncertainty is unchanged, but `return_informative` separates them
so a confidence meter does not reserve weight for a signal that said nothing.
"""
import numpy as np

from biocluster.annotation.annotator import compute_taxonomic_consistency


FULL = {
    "Apamea monoglypha": {"family": "Noctuidae"},
    "Mesapamea secalis": {"family": "Noctuidae"},
    "Chiasmia clathrata": {"family": "Geometridae"},
}
# common names parse to a genus but never a family
NAMES_ONLY = {
    "Horned Grebe": {"family": ""},
    "Eared Grebe": {"family": ""},
}


def test_pure_cluster_is_informative():
    cids = np.array([0, 0])
    tax, info = compute_taxonomic_consistency(
        cids, ["Apamea monoglypha", "Mesapamea secalis"], FULL,
        return_informative=True)
    assert tax.tolist() == [0.0, 0.0]        # one family -> pure
    assert info.tolist() == [True, True]     # and we know it


def test_mixed_cluster_scores_one_third():
    cids = np.array([0, 0])
    tax, info = compute_taxonomic_consistency(
        cids, ["Apamea monoglypha", "Chiasmia clathrata"], FULL,
        return_informative=True)
    assert np.allclose(tax, 1.0 / 3.0)
    assert info.all()


def test_unresolvable_cluster_scores_zero_but_is_not_informative():
    cids = np.array([0, 0])
    tax, info = compute_taxonomic_consistency(
        cids, ["Horned Grebe", "Eared Grebe"], NAMES_ONLY,
        return_informative=True)
    assert tax.tolist() == [0.0, 0.0]        # same score as a pure cluster
    assert info.tolist() == [False, False]   # but no evidence behind it


def test_noise_still_maxes_the_signal():
    """The floor that keeps unclustered images out of auto-accept must hold
    even when no family is resolvable."""
    cids = np.array([-1, 0, 0])
    tax, _ = compute_taxonomic_consistency(
        cids, ["Horned Grebe", "Horned Grebe", "Eared Grebe"], NAMES_ONLY,
        return_informative=True)
    assert tax[0] == 1.0


def test_empty_table_returns_both_arrays():
    cids = np.array([0, 0])
    tax, info = compute_taxonomic_consistency(
        cids, ["a", "b"], {}, return_informative=True)
    assert tax.tolist() == [0.0, 0.0]
    assert info.tolist() == [False, False]


def test_default_signature_is_unchanged():
    """Existing callers that do not ask for the flag still get one array."""
    cids = np.array([0, 0])
    out = compute_taxonomic_consistency(
        cids, ["Apamea monoglypha", "Mesapamea secalis"], FULL)
    assert isinstance(out, np.ndarray)
