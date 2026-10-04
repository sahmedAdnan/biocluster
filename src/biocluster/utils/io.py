"""
Utility functions for:
  - Setting global random seeds for reproducibility
  - Saving and loading embeddings, cluster results, and metrics
  - Config loading
"""

import json
import os
import random

import numpy as np
import torch
import yaml

def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducible results.

    Covers Python, NumPy, and PyTorch (CPU and GPU).

    GPU determinism note:
        CuBLAS on CUDA >= 10.2 requires an environment variable to be set
        *before* PyTorch initializes CUDA. We set it here programmatically,
        but if PyTorch was already initialized earlier in the process it will
        have no effect and you will see a UserWarning from open_clip.
        The warning means the operation used a fast non-deterministic
        kernel instead of a slower deterministic one. The difference is in
        floating-point rounding, but do not assume it is immaterial: on
        DK-Moth run01 a difference of 5e-06 per embedding value, well within
        rounding, moved the final ARI by 0.055 once UMAP and HDBSCAN had
        amplified it. If you intend to reproduce a published number, work from
        the saved embeddings rather than re-extracting.

        To silence the warning completely, set this environment variable
        in your terminal *before* running any Python command:

            Windows (Command Prompt):
                set CUBLAS_WORKSPACE_CONFIG=:4096:8

            Windows (PowerShell):
                $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

            Linux / macOS:
                export CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set the CuBLAS workspace config so deterministic mode works on CUDA >= 10.2.
        # This must be set before CUDA is first used. If CUDA was already initialized
        # (e.g. by importing open_clip before calling set_seed), the env var will have
        # no effect on this run, but it will work on the next clean run.
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Use warn_only=True so that operations without a deterministic
    # implementation (like some CuBLAS matmul kernels) print a warning
    # instead of raising an error and crashing the pipeline.
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[seed] All seeds set to {seed}")


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# -------------------------------------------------------
# Embeddings: Save and Load
# -------------------------------------------------------

def save_features(path: str, features: np.ndarray, labels: list,
                  paths: list, class_names: list) -> None:
    """
    Save extracted features and associated metadata to a .npz file.

    Args:
        path:        Full path to the output .npz file.
        features:    Array of shape (N, D).
        labels:      List of integer class IDs, length N.
        paths:       List of image file paths, length N.
        class_names: List of class name strings (ordered by class ID).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez(
        path,
        features=features,
        labels=np.array(labels, dtype=np.int32),
        paths=np.array(paths, dtype=object),
        class_names=np.array(class_names, dtype=object),
    )

    print(f"[save] Features saved to {path}  shape={features.shape}")


def load_features(path: str) -> tuple:
    """
    Load features from a .npz file.

    Returns:
        features, labels, paths, class_names
    """
    data = np.load(path, allow_pickle=True)

    features    = data["features"]
    labels      = data["labels"].tolist()
    paths       = data["paths"].tolist()
    class_names = data["class_names"].tolist()

    print(f"[load] Features loaded from {path}  shape={features.shape}")
    return features, labels, paths, class_names


# -------------------------------------------------------
# Cluster Results: Save and Load
# -------------------------------------------------------

def save_clusters(path: str, cluster_ids: np.ndarray, method_name: str,
                  params: dict) -> None:
    """
    Save cluster assignment IDs and the config used.

    Args:
        path:        Full path to the output .npz file.
        cluster_ids: Integer array of shape (N,).
        method_name: Name of the clustering method used.
        params:      The hyperparameter dict from the config.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez(
        path,
        cluster_ids=cluster_ids,
        method_name=np.array(method_name, dtype=object),
        params=np.array(json.dumps(params), dtype=object),
    )

    print(f"[save] Clusters saved to {path}")


def load_clusters(path: str) -> tuple:
    """
    Load cluster IDs and method metadata.

    Returns:
        cluster_ids, method_name, params_dict
    """
    data = np.load(path, allow_pickle=True)

    cluster_ids = data["cluster_ids"]
    method_name = str(data["method_name"])
    params      = json.loads(str(data["params"]))

    return cluster_ids, method_name, params


def save_metrics(path: str, metrics: dict) -> None:
    """Save metrics dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[save] Metrics saved to {path}")


def load_metrics(path: str) -> dict:
    """Load metrics from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------
# General
# -------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create a directory (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------------
# 2D Projection (shared across all CLI modules)
# -------------------------------------------------------

def get_2d_projection(features: np.ndarray, config: dict,
                      output_dir: str) -> np.ndarray:
    """
    Load a cached 2D UMAP/t-SNE projection or compute it fresh.

    The projection is used ONLY for visualization. Density-based methods
    (DBSCAN, HDBSCAN) receive PCA-reduced features as their actual input.

    The result is cached to disk so all CLI steps share the same projection.
    """
    cache_path = os.path.join(output_dir, "features_2d.npy")

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == features.shape[0]:
            print(f"  [proj] Loading cached 2D projection from {cache_path}")
            return cached
        else:
            print(f"  [proj] Cached projection has {cached.shape[0]} rows but "
                  f"features have {features.shape[0]} - recomputing.")

    vis = config["visualization"]
    method = vis.get("method", "umap")

    if method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            print("  [proj] umap-learn not installed - falling back to t-SNE.")
            method = "tsne"

    import time as _time

    if method == "umap":
        import umap as umap_lib
        cfg = vis.get("umap", {})   # absent block -> documented defaults below
        reducer = umap_lib.UMAP(
            n_neighbors  = cfg.get("n_neighbors", 30),
            n_components = 2,
            min_dist     = max(cfg.get("min_dist", 0.1), 0.05),
            metric       = cfg.get("metric", "cosine"),
            random_state = cfg.get("random_state", 42),
        )
        print(f"  [proj] Computing UMAP 2D projection on {features.shape[0]} "
              f"samples (this may take a few minutes) ...")
        _t0 = _time.time()
        f2d = reducer.fit_transform(features)
        print(f"  [proj] UMAP 2D done in {_time.time() - _t0:.1f}s")
    else:
        from sklearn.manifold import TSNE
        cfg = vis.get("tsne", {})
        reducer = TSNE(
            n_components  = 2,
            perplexity    = cfg.get("perplexity", 40),
            random_state  = cfg.get("random_state", 42),
            init          = "pca",
            learning_rate = "auto",
        )
        print(f"  [proj] Computing t-SNE 2D projection on {features.shape[0]} "
              f"samples (this may take a few minutes) ...")
        _t0 = _time.time()
        f2d = reducer.fit_transform(features)
        print(f"  [proj] t-SNE 2D done in {_time.time() - _t0:.1f}s")

    np.save(cache_path, f2d)
    print(f"  [proj] Saved to {cache_path}")
    return f2d
