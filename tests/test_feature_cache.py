"""The embedding cache: a fingerprint that depends on the right things."""
import numpy as np
import pytest

pytest.importorskip("PySide6", reason="feature_cache ships with the GUI extra")

from biocluster.app import feature_cache  # noqa: E402


def _img(tmp_path, name, content=b"x"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_fingerprint_is_stable_and_model_dependent(tmp_path):
    paths = [_img(tmp_path, "a.jpg"), _img(tmp_path, "b.jpg")]
    fp1 = feature_cache.fingerprint("bioclip2", paths)
    fp2 = feature_cache.fingerprint("bioclip2", paths)
    fp3 = feature_cache.fingerprint("clip", paths)
    assert fp1 == fp2
    assert fp1 != fp3


def test_fingerprint_changes_when_a_file_changes(tmp_path):
    paths = [_img(tmp_path, "a.jpg", b"one")]
    before = feature_cache.fingerprint("bioclip2", paths)
    (tmp_path / "a.jpg").write_bytes(b"a longer payload")
    assert feature_cache.fingerprint("bioclip2", paths) != before


def test_extraction_options_enter_the_key(tmp_path):
    paths = [_img(tmp_path, "a.jpg")]
    plain = feature_cache.fingerprint("dinov3", paths)
    pooled = feature_cache.fingerprint("dinov3", paths,
                                       opts={"global_pool": "token"})
    assert plain != pooled


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(feature_cache, "INDEX", str(tmp_path / "cache" / "index.json"))
    feats = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    feature_cache.save("deadbeef", feats, [0, 0, 1, 1, 2],
                       [f"{i}.jpg" for i in range(5)], ["a", "b", "c"])
    got = feature_cache.load("deadbeef")
    assert got is not None
    assert np.allclose(got[0], feats)
    assert "deadbeef" in feature_cache.entries()
