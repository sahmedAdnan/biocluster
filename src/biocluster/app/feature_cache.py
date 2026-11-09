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
format biocluster-extract writes. Files produced elsewhere are added with
import_npz.
"""

import hashlib
import json
import os
import re
import time

import numpy as np

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


def _write(fp: str, features, labels, image_paths, class_names, meta: dict = None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    from biocluster.utils.io import save_features
    save_features(path_for(fp), features,
                  list(labels) if labels is not None else [],
                  list(image_paths),
                  list(class_names) if class_names is not None else [],
                  model=(meta or {}).get("model"))
    _index_put(fp, meta or {}, features.shape)


def save(fp: str, features, labels, image_paths, class_names, meta: dict = None):
    """Write an entry plus its index record. Never raises -- a cache failure
    must not take down a run that already succeeded."""
    try:
        _write(fp, features, labels, image_paths, class_names, meta)
    except Exception as e:
        print(f"[cache] could not write entry {fp}: {e}")


def _match_keys(paths, depth: int):
    keys = []
    for p in paths:
        parts = [s for s in re.split(r"[\\/]+", str(p)) if s]
        keys.append("/".join(parts[-depth:]).casefold())
    return keys


def match_to_folder(file_paths, folder_paths) -> np.ndarray:
    """Return the row order of `file_paths` that aligns with `folder_paths`.

    Images are matched by class folder and file name, then by file name alone.
    Raises ValueError if the two lists do not contain the same images.
    """
    for depth in (2, 1):
        a, b = _match_keys(file_paths, depth), _match_keys(folder_paths, depth)
        if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
            continue
        pos = {k: i for i, k in enumerate(a)}
        return np.array([pos[k] for k in b], dtype=int)

    a, b = set(_match_keys(file_paths, 2)), _match_keys(folder_paths, 2)
    missing = [p for p, k in zip(folder_paths, b) if k not in a]
    extra = len(a - set(b))
    lines = ["The embeddings in this file do not belong to the images in this folder."]
    if missing:
        lines.append(f"{len(missing):,} image(s) in the folder have no embedding in the "
                     f"file, for example {os.path.basename(str(missing[0]))}.")
    if extra:
        lines.append(f"{extra:,} embedding(s) in the file have no image in the folder.")
    if not missing and not extra:
        lines.append("Some images share the same folder and file name, so they "
                     "cannot be told apart.")
    lines.append("Choose the folder the embeddings were made from, with no images "
                 "added or removed.")
    raise ValueError("\n".join(lines))


def _file_model(npz_path: str, data) -> str:
    """Model name recorded in the file, its extraction_meta.json, or a
    "<dataset>__<model>.npz" file name; None if absent."""
    if "model" in data.files:
        return str(data["model"])
    sidecar = os.path.join(os.path.dirname(npz_path), "extraction_meta.json")
    if os.path.basename(npz_path) == "features.npz" and os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as f:
                return json.load(f).get("model") or None
        except Exception:
            pass
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    return stem.split("__")[-1] if "__" in stem else None


def import_npz(npz_path: str, model: str, scans, opts: dict = None,
               expected_dim: int = None, known_models=(), meta: dict = None) -> dict:
    """Store an external embedding file as cache entries for a local folder.

    `scans` holds one (image_paths, labels, class_names) tuple per folder scan;
    each scan orders the images differently and gets its own entry.
    Returns {"n_images", "entries", "normalized"}. Raises ValueError on mismatch.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        features = np.asarray(data["features"], dtype=np.float32)
        file_paths = [str(p) for p in data["paths"].tolist()]
    except KeyError as e:
        raise ValueError(f"This is not a BioCluster embedding file: it has no {e} array.")
    except Exception as e:
        raise ValueError(f"Could not read the file: {e}")
    if features.ndim != 2 or len(features) != len(file_paths):
        raise ValueError("The file is damaged: it does not hold one embedding per image.")

    named = _file_model(npz_path, data)
    if named and named in known_models and named != model:
        raise ValueError(f"This file holds {named} embeddings, but the Model box is set "
                         f"to {model}. Choose {named} and import again.")
    if expected_dim and features.shape[1] != expected_dim:
        raise ValueError(f"These embeddings have {features.shape[1]} numbers per image, but "
                         f"{model} makes {expected_dim}. They come from a different model.")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = not np.allclose(norms, 1.0, atol=1e-3)
    if normalized:
        features = features / np.maximum(norms, 1e-12)

    scans = [s for s in scans if s and len(s[0])]
    if not scans:
        raise ValueError("No images found in the selected folder.")
    written, first_error = set(), None
    for image_paths, labels, class_names in scans:
        try:
            order = match_to_folder(file_paths, image_paths)
        except ValueError as e:
            first_error = first_error or e
            continue
        fp = fingerprint(model, image_paths, opts=opts)
        if fp in written:
            continue
        _write(fp, features[order], labels, image_paths, class_names,
               meta={**(meta or {}), "model": model,
                     "imported_from": os.path.abspath(npz_path),
                     "n_images": int(len(image_paths))})
        written.add(fp)
    if not written:
        raise first_error
    return {"n_images": int(len(features)), "entries": len(written), "normalized": bool(normalized)}


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
