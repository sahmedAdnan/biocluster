"""The embedding cache: fingerprinting, storage and import of external files."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="feature_cache ships with the GUI extra")

from biocluster.app import feature_cache  # noqa: E402
from biocluster.data.dataset import load_imagefolder  # noqa: E402
from biocluster.utils.io import save_features  # noqa: E402


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


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(feature_cache, "INDEX", str(tmp_path / "cache" / "index.json"))


@pytest.fixture()
def labelled_folder(tmp_path):
    root = tmp_path / "collection"
    for species, names in {"Genus alpha": ["a1.jpg", "a2.jpg"],
                           "Genus beta": ["b1.jpg", "b2.jpg", "b3.jpg"]}.items():
        (root / species).mkdir(parents=True)
        for n in names:
            (root / species / n).write_bytes(n.encode())
    paths, labels, classes = load_imagefolder(str(root), ["jpg"])
    feats = np.random.default_rng(0).normal(size=(len(paths), 8)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return paths, labels, classes, feats


def _remote_paths(paths, order):
    return ["/mnt/remote/collection/" + "/".join(Path(paths[i]).parts[-2:]) for i in order]


def test_match_by_class_folder_and_file_name():
    in_file = [r"E:\data\Genus beta\b2.jpg", "/home/x/data/Genus alpha/a1.jpg",
               r"E:\data\Genus beta\b1.jpg"]
    in_folder = ["D:/collection\\Genus alpha\\a1.jpg", "D:/collection\\Genus beta\\b1.jpg",
                 "D:/collection\\Genus beta\\b2.jpg"]
    assert list(feature_cache.match_to_folder(in_file, in_folder)) == [1, 2, 0]


def test_match_falls_back_to_file_names():
    in_folder = ["D:/collection\\Genus alpha\\a1.jpg", "D:/collection\\Genus beta\\b1.jpg"]
    assert list(feature_cache.match_to_folder(["b1.jpg", "a1.jpg"], in_folder)) == [1, 0]


def test_match_rejects_different_images():
    in_folder = ["D:/collection\\Genus alpha\\a1.jpg", "D:/collection\\Genus beta\\b1.jpg"]
    with pytest.raises(ValueError, match="no embedding"):
        feature_cache.match_to_folder(["/x/Genus alpha/a1.jpg"], in_folder)


def test_import_writes_the_pipeline_cache_entry(tmp_path, cache, labelled_folder):
    paths, labels, classes, feats = labelled_folder
    order = np.random.default_rng(1).permutation(len(paths))
    npz = tmp_path / "collection__bioclip2.5.npz"
    save_features(str(npz), feats[order], [0] * len(paths), _remote_paths(paths, order), classes)

    info = feature_cache.import_npz(str(npz), "bioclip2.5", [(paths, labels, classes)],
                                    expected_dim=8, known_models=("bioclip2.5", "dinov2"))
    assert info == {"n_images": 5, "entries": 1, "normalized": False}
    got = feature_cache.load(feature_cache.fingerprint("bioclip2.5", paths))
    assert got is not None
    assert np.allclose(got[0], feats)
    assert got[1] == list(labels)


def test_import_rejects_another_model(tmp_path, cache, labelled_folder):
    paths, labels, classes, feats = labelled_folder
    scans = [(paths, labels, classes)]
    named = tmp_path / "named.npz"
    save_features(str(named), feats, labels, paths, classes, model="dinov2")
    with pytest.raises(ValueError, match="dinov2"):
        feature_cache.import_npz(str(named), "bioclip2.5", scans, expected_dim=8,
                                 known_models=("bioclip2.5", "dinov2"))
    unnamed = tmp_path / "unnamed.npz"
    save_features(str(unnamed), feats, labels, paths, classes)
    with pytest.raises(ValueError, match="numbers per image"):
        feature_cache.import_npz(str(unnamed), "bioclip2.5", scans, expected_dim=1024)


def test_import_normalises_embeddings(tmp_path, cache, labelled_folder):
    paths, labels, classes, feats = labelled_folder
    npz = tmp_path / "raw.npz"
    save_features(str(npz), feats * 3.0, labels, paths, classes)
    info = feature_cache.import_npz(str(npz), "bioclip2.5", [(paths, labels, classes)])
    assert info["normalized"]
    got = feature_cache.load(feature_cache.fingerprint("bioclip2.5", paths))
    assert np.allclose(np.linalg.norm(got[0], axis=1), 1.0)
