"""
Unified clustering API.

All methods accept:
  - features:     np.ndarray of shape (N, D)
  - params:       dict from the YAML config block for this method
  - extra kwargs: optional

All methods return:
  - cluster_ids:  np.ndarray of shape (N,), integer cluster assignments

Supported methods (set via config clustering.methods[].name):
  - kmeans
  - spectral
  - agglomerative
  - dbscan
  - hdbscan      (requires: pip install hdbscan)
  - gmm
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

# -------------------------------------------------------
# K-Means (Baseline)
# -------------------------------------------------------

def run_kmeans(features: np.ndarray, params: dict, **kwargs) -> np.ndarray:
    """
    Standard K-Means with random initialization.

    Relevant params:
        n_clusters:   number of clusters (required)
        n_init:       number of restarts (default 50)
        max_iter:     max iterations per run (default 300)
        random_state: random seed (default 42)
    """
    km = KMeans(
        n_clusters   = params["n_clusters"],
        n_init       = params.get("n_init", 50),
        max_iter     = params.get("max_iter", 300),
        random_state = params.get("random_state", 42),
    )

    cluster_ids = km.fit_predict(features)
    print(f"  [kmeans] {params['n_clusters']} clusters, "
          f"inertia={km.inertia_:.2f}")

    return cluster_ids


def run_spectral(features: np.ndarray, params: dict, **kwargs) -> np.ndarray:
    """
    Spectral Clustering.

    Note: Spectral Clustering builds an affinity matrix (O(N^2) memory).
    For large datasets (N > 5000), consider subsampling first.

    The "Graph is not fully connected" warning from sklearn means some images
    are isolated from the main similarity graph, usually because n_neighbors is
    too small relative to the dataset size. Increasing n_neighbors (15-20 for
    ~2000 images) resolves this. The warning is suppressed below since we
    document and handle it here: the result is still valid, but the user
    should know the graph connectivity affects result quality.

    Relevant params:
        n_clusters, n_init, random_state
        n_neighbors: number of neighbors for affinity graph (default 15)
    """
    import warnings

    sc = SpectralClustering(
        n_clusters   = params["n_clusters"],
        n_init       = params.get("n_init", 10),
        affinity     = "nearest_neighbors",
        n_neighbors  = params.get("n_neighbors", 15),  # 15 prevents disconnected graphs for ~2000 images
        random_state = params.get("random_state", 42),
        n_jobs       = -1,
    )

    # Suppress the "not fully connected" sklearn warning.
    # It is a known behavior with nearest-neighbor affinity when n_neighbors
    # is small. We use n_neighbors=15 by default to minimise this risk.
    # If the warning still appears, increase n_neighbors in the config.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message="Graph is not fully connected",
            category=UserWarning)
        cluster_ids = sc.fit_predict(features)

    print(f"  [spectral] {params['n_clusters']} clusters, done.")

    return cluster_ids


# -------------------------------------------------------
# Agglomerative / Hierarchical Clustering
# -------------------------------------------------------

def run_agglomerative(features: np.ndarray, params: dict, **kwargs) -> np.ndarray:
    """
    Agglomerative (hierarchical) clustering using Ward linkage by default.

    Ward minimizes the variance within each cluster at each merge step.
    It tends to produce compact, even-sized clusters.

    Relevant params:
        n_clusters, linkage ('ward'|'complete'|'average'|'single')
    """
    agg = AgglomerativeClustering(
        n_clusters = params["n_clusters"],
        linkage    = params.get("linkage", "ward"),
        # random_state not supported: AgglomerativeClustering is deterministic
    )

    cluster_ids = agg.fit_predict(features)
    print(f"  [agglomerative] {params['n_clusters']} clusters "
          f"(linkage={params.get('linkage','ward')}), done.")

    return cluster_ids


# -------------------------------------------------------
# DBSCAN (density-based, discovers number of clusters)
# -------------------------------------------------------

def run_dbscan(features: np.ndarray, params: dict, **kwargs) -> np.ndarray:
    """
    DBSCAN: density-based clustering.

    Does NOT require n_clusters. The number of clusters is discovered
    from the data. Points that do not belong to any cluster are labelled -1.

    Relevant params:
        eps:         neighborhood radius
        min_samples: minimum neighbors to form a core point
    """
    db = DBSCAN(
        eps         = params.get("eps", 0.5),
        min_samples = params.get("min_samples", 5),
        metric      = "euclidean",
        n_jobs      = -1,
        # random_state not supported: DBSCAN is deterministic
    )

    cluster_ids = db.fit_predict(features)

    n_found  = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise  = int((cluster_ids == -1).sum())

    print(f"  [dbscan] {n_found} clusters found, "
          f"{n_noise} noise points (eps={params.get('eps',0.5)})")

    return cluster_ids


def run_hdbscan(features: np.ndarray, params: dict,
                return_probabilities: bool = False, **kwargs):
    """
    HDBSCAN: hierarchical density-based clustering.

    More robust than DBSCAN because it adapts the density threshold
    per region rather than using a global eps.

    Requires: pip install hdbscan

    Relevant params:
        min_cluster_size: minimum number of points in a cluster (default 5)
        min_samples:      core point requirement (default None, uses min_cluster_size)

    If return_probabilities is True, also returns the per-point membership
    probability (0 for noise). This is the spatial uncertainty signal used by
    the annotation step; without it, annotation falls back to silhouette.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("HDBSCAN requires: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size          = params.get("min_cluster_size", 5),
        min_samples               = params.get("min_samples", None),
        metric                    = "euclidean",
        cluster_selection_method  = params.get("cluster_selection_method", "eom"),
        cluster_selection_epsilon = float(params.get("cluster_selection_epsilon", 0.0)),
        # random_state not supported: HDBSCAN is deterministic
    )

    cluster_ids = clusterer.fit_predict(features)

    n_found = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = int((cluster_ids == -1).sum())

    print(f"  [hdbscan] {n_found} clusters found, {n_noise} noise points")

    if return_probabilities:
        return cluster_ids, clusterer.probabilities_
    return cluster_ids


# -------------------------------------------------------
# Gaussian Mixture Models
# -------------------------------------------------------

def run_gmm(features: np.ndarray, params: dict, **kwargs) -> np.ndarray:
    """
    Gaussian Mixture Model (soft clustering, hard-assigned here).

    Automatically retries with increasing reg_covar if the covariance matrix
    becomes ill-conditioned (common on UMAP-reduced features where some
    components may have very few or collapsed points).

    Relevant params:
        n_clusters:       number of Gaussian components (required)
        covariance_type:  'full' | 'tied' | 'diag' | 'spherical' (default 'full')
        reg_covar:        regularization added to covariance diagonal (default 1e-6).
                          Auto-escalated to 1e-4 -> 1e-3 -> 1e-2 -> 1e-1 on failure.
        random_state:     random seed (default 42)
        max_iter:         max EM iterations (default 200)
    """
    base_reg     = params.get("reg_covar", 1e-6)
    cov_type     = params.get("covariance_type", "full")
    random_state = params.get("random_state", 42)
    max_iter     = params.get("max_iter", 200)
    n_components = params["n_clusters"]

    retry_regs = sorted(set([base_reg, 1e-4, 1e-3, 1e-2, 1e-1]))

    last_error = None
    for reg in retry_regs:
        try:
            gmm = GaussianMixture(
                n_components    = n_components,
                covariance_type = cov_type,
                reg_covar       = reg,
                random_state    = random_state,
                max_iter        = max_iter,
            )
            cluster_ids = gmm.fit_predict(features)
            if reg > base_reg:
                print(f"  [gmm] reg_covar escalated to {reg:.0e} for numerical stability")
            print(f"  [gmm] {n_components} components, BIC={gmm.bic(features):.2f}")
            return cluster_ids
        except ValueError as e:
            last_error = e
            continue

    raise ValueError(
        f"GMM failed on all reg_covar values {retry_regs}. "
        f"The features may be too non-Gaussian for GMM (e.g. UMAP embeddings "
        f"with many components relative to sample size). "
        f"Consider using 'diag' or 'spherical' covariance_type, or skip GMM "
        f"for this reduction. Original error: {last_error}"
    )


# -------------------------------------------------------
# Dispatcher: Run Any Method by Name
# -------------------------------------------------------

METHOD_MAP = {
    "kmeans":             run_kmeans,
    "spectral":           run_spectral,
    "agglomerative":      run_agglomerative,
    "dbscan":             run_dbscan,
    "hdbscan":            run_hdbscan,
    "gmm":                run_gmm,
}


def run_clustering(method_name: str, features: np.ndarray,
                   params: dict, **kwargs) -> np.ndarray:
    """
    Run a clustering method by name.

    Args:
        method_name: One of the keys in METHOD_MAP.
        features:    Feature array (N, D).
        params:      Hyperparameter dict from config.
        **kwargs:    Extra args passed to the method.

    Returns:
        cluster_ids: Integer array (N,).
    """
    if method_name not in METHOD_MAP:
        raise ValueError(
            f"Unknown clustering method: '{method_name}'. "
            f"Available: {list(METHOD_MAP.keys())}"
        )

    print(f"\n[clustering] Running: {method_name}")
    cluster_ids = METHOD_MAP[method_name](features, params, **kwargs)

    return cluster_ids
