"""
app/feature_cache.py
--------------------
On-disk cache for extracted embeddings.

Extraction is by far the slowest step in the pipeline and its result depends on
only three things: which images, which model, and whether embeddings are
normalised. Reduction, clustering, seed and n_clusters do not affect it at all
-- so once a folder has been extracted with a given model, every later
experiment on that folder should be instant.

Before this module the app cached features in a dict on MainWindow, which died
with the process, so every launch re-extracted from scratch. That cache was also
keyed on batch_size and device, so nudging the batch size silently threw away a
perfectly good cache.

Layout
------
    ~/BioCluster/feature_cache/
        index.json          human-readable list of what is cached
        <fingerprint>.npz   features + labels + paths + class names

The .npz files use biocluster.utils.io.save_features / load_features, the same
format biocluster-extract writes, so a features.npz produced on a rented cloud
GPU can be dropped straight in.
"""

import hashlib
import json
import os
import time

CACHE_DIR = os.path.join(os.path.expanduser("~"), "BioCluster", "feature_cache")
INDEX     = os.path.join(CACHE_DIR, "index.json")


def fingerprint(model: str, image_paths, normalize: bool = True,
                opts: dict = None) -> str:
    """Identity of an extraction result.

    Covers the model, the exact file list, and each file's size and mtime, so
    editing, adding or removing an image invalidates the entry.

    batch_size and device are deliberately excluded, but not because they make
    no difference. They do not change the computation, yet they do change its
    floating-point result, because the GPU kernel chosen depends on the batch
    shape. Measured on DK-Moth run01 with BioCLIP 2.5, batch_size 32 against 64
    moved individual values by up to 5e-06, and after UMAP and HDBSCAN that
    shifted ARI from 0.785 to 0.830. They stay out of the key so that changing
    a batch size for speed does not throw away usable embeddings, which is the
    right trade; the consequence is that a cache hit is not a promise of
    bit-identical features, so a published number should be reproduced from the
    saved .npz rather than by re-extracting.

    `opts` carries any model setting that DOES change the output -- currently
    global_pool and img_size. This is not optional bookkeeping: two DINOv3 runs
    differing only in global_pool would otherwise share a key, so the second
    would silently load the first's embeddings and you would measure "pooling
    makes no difference" with no error to reveal it.
    """
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"|norm=" + (b"1" if normalize else b"0"))
    for k in sorted(opts or {}):
        v = (opts or {})[k]
        if v not in (None, ""):
            h.update(f"|{k}={v}".encode("utf-8"))
    for p in image_paths:                     # already sorted by the caller
        try:
            st = os.stat(p)
            stamp = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            stamp = f"{p}|missing"
        h.update(stamp.encode("utf-8", "replace"))
    return h.hexdigest()[:32]


def path_for(fp: str) -> str:
    return os.path.join(CACHE_DIR, f"{fp}.npz")


def load(fp: str):
    """Return (features, labels, paths, class_names) or None on any miss."""
    p = path_for(fp)
    if not os.path.exists(p):
        return None
    try:
        from biocluster.utils.io import load_features
        return load_features(p)
    except Exception as e:
        print(f"[cache] ignoring unreadable entry {fp}: {e}")
        return None


def save(fp: str, features, labels, image_paths, class_names, meta: dict = None):
    """Write an entry plus its index record. Never raises -- a cache failure
    must not take down a run that already succeeded."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        from biocluster.utils.io import save_features
        save_features(path_for(fp), features,
                      list(labels) if labels is not None else [],
                      list(image_paths),
                      list(class_names) if class_names is not None else [])
        _index_put(fp, meta or {}, features.shape)
    except Exception as e:
        print(f"[cache] could not write entry {fp}: {e}")


def _read_index() -> dict:
    try:
        with open(INDEX, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _index_put(fp: str, meta: dict, shape) -> None:
    idx = _read_index()
    idx[fp] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape":   list(shape),
        "size_mb": round(os.path.getsize(path_for(fp)) / 1e6, 1),
        **meta,
    }
    tmp = INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)
    os.replace(tmp, INDEX)


def entries() -> dict:
    """Everything currently cached, newest first."""
    return dict(sorted(_read_index().items(),
                       key=lambda kv: kv[1].get("created", ""), reverse=True))


def total_size_mb() -> float:
    if not os.path.isdir(CACHE_DIR):
        return 0.0
    return round(sum(os.path.getsize(os.path.join(CACHE_DIR, f))
                     for f in os.listdir(CACHE_DIR)
                     if f.endswith(".npz")) / 1e6, 1)


def clear() -> int:
    """Delete every cached entry. Returns how many were removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    n = 0
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".npz"):
            try:
                os.remove(os.path.join(CACHE_DIR, f)); n += 1
            except OSError:
                pass
    try:
        os.remove(INDEX)
    except OSError:
        pass
    return n
