"""
Full DBSCAN diagnostic, tuning, and evaluation pipeline.

Feature space:
    The caller (biocluster-cluster) reduces the features and passes them in
    via features_reduced. UMAP is used for the reported runs: it preserves
    the local neighbourhood structure DBSCAN depends on for density
    estimation. On EU-Moth this gave ARI 0.9798 and on CUB-200 ARI 0.8455.

    If no pre-reduced features are passed, the pipeline applies its own PCA
    instead, which lets it be driven directly in a notebook.

    eps is an absolute radius in whatever space it receives, so a value is
    only valid for the reduction it was derived under. See
    scripts/pick_dbscan_eps.py.

Noise handling:
    All metrics (ARI, NMI, accuracy) exclude noise points (label=-1).
    Coverage-weighted ARI = ARI × (1 - noise_pct) is used for eps selection.

Usage:
    from biocluster.clustering.density.dbscan_pipeline import DBSCANPipeline
    pipeline = DBSCANPipeline(config)
    results  = pipeline.run(features, true_label_ids, class_names,
                            features_2d, output_dir,
                            features_reduced=umap_features,
                            reduction_label="UMAP-100D")
"""

import json
import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")

from biocluster.evaluation.metrics import (
    hungarian_accuracy,
    per_class_accuracy,
    top_k_confusion_pairs,
)

# -------------------------------------------------------
# Style constants, consistent across all density figures
# -------------------------------------------------------

STYLE = {
    "font_family":   "sans-serif",
    "title_size":    13,
    "label_size":    11,
    "tick_size":     9,
    "legend_size":   9,
    "dpi":           300,
    "noise_color":   "#AAAAAA",   # medium grey for noise points
    "noise_label":   "Noise (-1)",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family":    STYLE["font_family"],
    "axes.titlesize": STYLE["title_size"],
    "axes.labelsize": STYLE["label_size"],
    "xtick.labelsize": STYLE["tick_size"],
    "ytick.labelsize": STYLE["tick_size"],
    "legend.fontsize": STYLE["legend_size"],
})


# -------------------------------------------------------
# Helper: Save figure as PNG (+PDF if requested)
# -------------------------------------------------------

def _save(fig: plt.Figure, output_dir: str, stem: str,
          export_pdf: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(png, dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    if export_pdf:
        fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  [fig] {png}")


# -------------------------------------------------------
# Step 0: Feature space preparation
# -------------------------------------------------------

def prepare_features(features: np.ndarray,
                     n_pca_components: int = 100) -> Tuple[np.ndarray, object]:
    """
    Reduce features with PCA.

    Only used when the caller passes no features_reduced. The reported runs
    all reduce with UMAP in biocluster.reduction and pass the result in, so
    this path exists for driving the pipeline directly.

    Args:
        features:         Raw (L2-normalized) feature array, shape (N, D).
        n_pca_components: Number of PCA dimensions to retain.

    Returns:
        (reduced_features, fitted_pca_object)
    """
    print(f"  [prep] Applying PCA: {features.shape[1]}D -> {n_pca_components}D ...")
    pca = PCA(n_components=n_pca_components, random_state=42)
    reduced = pca.fit_transform(features)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  [prep] Variance retained: {variance_explained:.1%}")
    print(f"  [prep] Reduced shape: {reduced.shape}")

    return reduced, pca


# -------------------------------------------------------
# Step 1: k-distance plot (eps guidance)
# -------------------------------------------------------

def plot_kdistance(features: np.ndarray, min_samples: int,
                   output_dir: str, export_pdf: bool = False) -> float:
    """
    Compute and plot the k-distance graph to guide eps selection.

    HOW TO READ THIS PLOT:
        For each point, compute the distance to its k-th nearest neighbor
        (k = min_samples). Sort these distances in ascending order and plot.
        The "knee" (elbow) in this curve is the recommended eps value.

        Points to the right of the knee will be treated as noise.
        Points to the left will be core points. If the knee is not sharp,
        DBSCAN will be sensitive to the exact eps value (a known limitation).

    Args:
        features:    Feature array to compute distances in.
        min_samples: The min_samples you plan to use in DBSCAN.
        output_dir:  Where to save the plot.
        export_pdf:  Also save a PDF.

    Returns:
        Detected knee value (float): suggested eps.
    """
    print(f"  [k-dist] Computing {min_samples}-NN distances ...")

    nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean",
                            n_jobs=-1)
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)

    # The k-th neighbor distance for each point (column index min_samples-1)
    k_distances = np.sort(distances[:, min_samples - 1])[::-1]

    # Smooth the curve slightly for knee detection
    if len(k_distances) > 51:
        window = min(51, len(k_distances) // 20 * 2 + 1)   # must be odd
        smoothed = savgol_filter(k_distances, window_length=window, polyorder=3)
    else:
        smoothed = k_distances

    # Knee detection: maximum curvature point on the sorted distance curve
    # We use the "kneedle" concept: find the point with maximum distance
    # from the line connecting the first and last points of the curve.
    x = np.arange(len(smoothed))
    p1 = np.array([x[0],  smoothed[0]])
    p2 = np.array([x[-1], smoothed[-1]])
    line_vec  = p2 - p1
    line_unit = line_vec / np.linalg.norm(line_vec)

    # Distance of each point from the line p1->p2
    proj      = np.outer(np.dot(np.column_stack([x, smoothed]) - p1, line_unit),
                         line_unit) + p1
    distances_from_line = np.linalg.norm(
        np.column_stack([x, smoothed]) - proj, axis=1
    )
    knee_idx  = int(np.argmax(distances_from_line))
    knee_eps  = float(k_distances[knee_idx])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(k_distances, color="#4C72B0", linewidth=1.4, label="k-distance (sorted)")
    ax.axvline(x=knee_idx, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Knee — suggested eps = {knee_eps:.3f}")
    ax.axhline(y=knee_eps, color="#C44E52", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Points (sorted by decreasing k-distance)")
    ax.set_ylabel(f"Distance to {min_samples}-th nearest neighbor")
    ax.set_title(f"k-Distance Plot  (k = {min_samples})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Caption as figure text below the axes
    fig.text(
        0.5, -0.04,
        "The knee of this curve is the recommended eps value for DBSCAN. "
        "Points to the right of the knee become noise.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    _save(fig, output_dir, "dbscan_kdistance_plot", export_pdf)
    print(f"  [k-dist] Detected knee at eps = {knee_eps:.4f}")

    return knee_eps


def eps_sweep(features: np.ndarray, true_label_ids: np.ndarray,
              eps_values: np.ndarray, min_samples: int,
              output_dir: str, export_pdf: bool = False,
              n_true_classes: int = None) -> pd.DataFrame:
    """
    Sweep over eps values and record: ARI, NMI, % noise, # clusters.

    IMPORTANT: NOISE HANDLING IN METRICS:
        All metrics (ARI, NMI) are computed AFTER excluding noise points.
        Including noise points (label=-1) creates a phantom cluster that
        massively distorts ARI by grouping thousands of images from hundreds
        of species into one fake "cluster -1". The true clustering quality
        of the non-noise points is always reported separately.

    Args:
        features:       PCA-reduced feature array.
        true_label_ids: Integer ground-truth class IDs, shape (N,).
        eps_values:     1D array of eps values to try.
        min_samples:    Fixed min_samples for all runs.
        output_dir:     Where to save the sweep plot.
        export_pdf:     Also save PDF.

    Returns:
        DataFrame with one row per eps value and columns:
        eps, n_clusters, noise_pct, ari, nmi
    """
    print(f"  [sweep] eps sweep: {len(eps_values)} values, "
          f"min_samples={min_samples} ...")

    rows = []

    for j, eps in enumerate(eps_values, 1):
        db     = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                        n_jobs=-1)
        labels = db.fit_predict(features)

        n_total    = len(labels)
        n_noise    = int((labels == -1).sum())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = n_noise / n_total

        # Exclude noise before computing metrics
        valid      = labels >= 0
        if valid.sum() > 0 and n_clusters > 1:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The number of unique classes",
                                        category=UserWarning)
                ari = adjusted_rand_score(true_label_ids[valid], labels[valid])
                nmi = normalized_mutual_info_score(true_label_ids[valid], labels[valid])
        else:
            ari = 0.0
            nmi = 0.0

        # Coverage-weighted score: ARI * (1 - noise_pct)
        #
        # WHY NOT ARI ALONE:
        #   A degenerate DBSCAN with eps too small finds tiny perfectly-pure
        #   micro-clusters and labels everything else noise, giving ARI=1.0
        #   on 3% of the data while discarding 97%. Raw ARI cannot detect
        #   this failure. Multiplying by coverage (1 - noise_pct) penalises
        #   solutions that achieve purity only by discarding most of the data.
        #   ARI=1.0 with 97% noise  ->  score = 1.0 * 0.03 = 0.03  (bad)
        #   ARI=0.97 with 2% noise  ->  score = 0.97 * 0.98 = 0.95 (good)
        coverage       = 1.0 - noise_pct
        weighted_score = ari * coverage

        print(f"    [{j:2d}/{len(eps_values)}] eps={eps:.4f} -> "
              f"clusters={n_clusters:4d}, noise={noise_pct:.1%}, "
              f"ARI={ari:.4f}, score={weighted_score:.4f}")

        rows.append(dict(eps=eps, n_clusters=n_clusters,
                         noise_pct=noise_pct, ari=ari, nmi=nmi,
                         coverage=coverage, weighted_score=weighted_score))

    df = pd.DataFrame(rows)

    # --- Sweep plot: 4 panels ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"eps Sweep  (min_samples = {min_samples})",
                 fontsize=14, y=1.02)

    # Mark the best eps by weighted score on all panels
    best_idx = df["weighted_score"].idxmax()
    best_eps_val = df.loc[best_idx, "eps"]

    # Panel 1: ARI and NMI vs eps
    axes[0].plot(df["eps"], df["ari"], color="#4C72B0", linewidth=2,
                 marker="o", markersize=4, label="ARI")
    axes[0].plot(df["eps"], df["nmi"], color="#55A868", linewidth=2,
                 marker="s", markersize=4, label="NMI", linestyle="--")
    axes[0].axvline(best_eps_val, color="#C44E52", linestyle="--",
                    linewidth=1.5, label=f"Best eps={best_eps_val:.3f}")
    axes[0].set_xlabel("eps")
    axes[0].set_ylabel("Score (excl. noise)")
    axes[0].set_title("ARI / NMI vs eps")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: % noise vs eps
    axes[1].fill_between(df["eps"], df["noise_pct"] * 100,
                         alpha=0.35, color="#C44E52")
    axes[1].plot(df["eps"], df["noise_pct"] * 100,
                 color="#C44E52", linewidth=2, marker="o", markersize=4)
    axes[1].axvline(best_eps_val, color="#C44E52", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("eps")
    axes[1].set_ylabel("% Noise Points")
    axes[1].set_title("Noise Fraction vs eps")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: # clusters vs eps
    axes[2].plot(df["eps"], df["n_clusters"], color="#8172B2",
                 linewidth=2, marker="^", markersize=4)
    _true_k = n_true_classes if n_true_classes is not None else 200
    axes[2].axhline(y=_true_k, color="#FF7F0E", linestyle="--", linewidth=1.5,
                    label=f"True # classes ({_true_k})")
    axes[2].axvline(best_eps_val, color="#C44E52", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("eps")
    axes[2].set_ylabel("Clusters Found")
    axes[2].set_title("Clusters Found vs eps")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Coverage-weighted score, the actual selection criterion
    # This is ARI * (1 - noise_pct). It penalises high-ARI runs that
    # discard most of the data as noise. The peak of this curve is used
    # to select the final eps automatically.
    axes[3].plot(df["eps"], df["weighted_score"], color="#D55E00",
                 linewidth=2.5, marker="D", markersize=5,
                 label="ARI × Coverage")
    axes[3].axvline(best_eps_val, color="#C44E52", linestyle="--",
                    linewidth=1.5,
                    label=f"Selected eps={best_eps_val:.3f}")
    axes[3].set_xlabel("eps")
    axes[3].set_ylabel("ARI x (1 - noise %)")
    axes[3].set_title("Selection Score (ARI x Coverage)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.text(
        0.5, -0.06,
        "Selection criterion: ARI × Coverage, where Coverage = 1 − noise_fraction. "
        "This penalises degenerate solutions that achieve high ARI by discarding "
        "most of the data as noise. The vertical dashed line marks the selected eps.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "dbscan_eps_sweep", export_pdf)

    return df


# -------------------------------------------------------
# Step 3: Final DBSCAN fit and full evaluation
# -------------------------------------------------------

def fit_and_evaluate(features: np.ndarray, true_label_ids: np.ndarray,
                     class_names: List[str],
                     eps: float, min_samples: int,
                     top_k: int = 20) -> dict:
    """
    Fit DBSCAN with the chosen parameters and compute all metrics.

    Noise handling:
        - Noise points are excluded from ARI, NMI, and Hungarian accuracy.
        - Noise count and percentage are reported explicitly.
        - Noise points are visible in all plots (grey, distinct marker).

    Returns a dict with all results.
    """
    import time as _time
    print(f"\n  [dbscan] Final fit: eps={eps:.4f}, min_samples={min_samples} ...")
    _t0 = _time.time()

    db     = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                    n_jobs=-1)
    labels = db.fit_predict(features)
    print(f"  [dbscan] Fit completed in {_time.time() - _t0:.1f}s")

    n_total    = len(labels)
    n_noise    = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"  [dbscan] Clusters found: {n_clusters}")
    print(f"  [dbscan] Noise points:   {n_noise} ({n_noise/n_total:.1%})")

    # Compute metrics on non-noise points only
    valid      = labels >= 0
    t_valid    = true_label_ids[valid]
    c_valid    = labels[valid]

    if valid.sum() > 0 and n_clusters > 1:
        ari     = float(adjusted_rand_score(t_valid, c_valid))
        nmi     = float(normalized_mutual_info_score(t_valid, c_valid))
        acc, mapping = hungarian_accuracy(t_valid, c_valid)
        per_cls_df   = per_class_accuracy(t_valid, c_valid, mapping, class_names)
        pred_labels  = mapping[c_valid]
        conf_df      = top_k_confusion_pairs(t_valid, pred_labels,
                                             class_names, k=top_k)
    else:
        ari = nmi = acc = 0.0
        mapping     = np.zeros(max(1, labels.max() + 2), dtype=np.int64)
        per_cls_df  = pd.DataFrame()
        conf_df     = pd.DataFrame()

    print(f"  [dbscan] ARI (excl. noise): {ari:.4f}")
    print(f"  [dbscan] NMI (excl. noise): {nmi:.4f}")
    print(f"  [dbscan] Acc (excl. noise): {acc:.4f}")

    return {
        "method":      "dbscan",
        "eps":         eps,
        "min_samples": min_samples,
        "n_clusters":  n_clusters,
        "n_noise":     n_noise,
        "noise_pct":   n_noise / n_total,
        "ari":         ari,
        "nmi":         nmi,
        "accuracy":    float(acc),
        "labels":      labels,
        "mapping":     mapping.tolist(),
        "per_class_df": per_cls_df,
        "conf_df":     conf_df,
    }


# -------------------------------------------------------
# Step 4: Publication-quality visualizations
# -------------------------------------------------------

def plot_umap_dbscan(features_2d: np.ndarray, dbscan_labels: np.ndarray,
                     true_label_ids: np.ndarray,
                     results: dict,
                     output_dir: str, export_pdf: bool = False,
                     subsample: int = 6000) -> None:
    """
    Side-by-side scatter: true labels vs DBSCAN clusters.

    Noise points (label=-1) are shown in grey with a distinct marker shape
    so they are never hidden or confused with a real cluster.

    With 200 classes, individual colors are not distinguishable.
    A smooth colormap is used and the caption explains this clearly.
    """
    n = len(dbscan_labels)
    rng = np.random.default_rng(42)

    # Subsample for readability (scatter overdraw is misleading at full N)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
    else:
        idx = np.arange(n)

    f2d   = features_2d[idx]
    dlab  = dbscan_labels[idx]
    tlab  = true_label_ids[idx]

    n_clusters = results["n_clusters"]
    noise_pct  = results["noise_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("DBSCAN Clustering on BioCLIP-2 Embeddings (EU-Moth)",
                 fontsize=14, y=1.01)

    # ---- Panel A: True species labels ----
    cmap_true = plt.cm.nipy_spectral
    axes[0].scatter(
        f2d[:, 0], f2d[:, 1],
        c=tlab, cmap=cmap_true,
        s=6, alpha=0.5, linewidths=0, rasterized=True
    )
    axes[0].set_title("A)  Ground-Truth Species Labels")
    axes[0].set_xlabel("UMAP Component 1")
    axes[0].set_ylabel("UMAP Component 2")
    axes[0].set_aspect("equal", "datalim")

    # ---- Panel B: DBSCAN cluster assignments ----
    # Noise points drawn separately in grey with a different marker
    noise_mask  = dlab == -1
    valid_mask  = ~noise_mask

    if valid_mask.sum() > 0:
        axes[1].scatter(
            f2d[valid_mask, 0], f2d[valid_mask, 1],
            c=dlab[valid_mask], cmap=plt.cm.nipy_spectral,
            s=6, alpha=0.5, linewidths=0, rasterized=True,
            label=f"Clusters ({n_clusters})"
        )

    if noise_mask.sum() > 0:
        axes[1].scatter(
            f2d[noise_mask, 0], f2d[noise_mask, 1],
            c=STYLE["noise_color"], marker="x", s=12, alpha=0.6,
            linewidths=0.6, label=f"Noise ({noise_pct:.1%})"
        )

    axes[1].set_title(
        f"B)  DBSCAN Clusters  "
        f"(ARI={results['ari']:.3f}, "
        f"Acc={results['accuracy']:.3f}, "
        f"noise={noise_pct:.1%})"
    )
    axes[1].set_xlabel("UMAP Component 1")
    axes[1].set_ylabel("UMAP Component 2")
    axes[1].set_aspect("equal", "datalim")
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.85,
                   markerscale=2.5, fontsize=8)

    fig.text(
        0.5, -0.03,
        "Embeddings projected to 2D with UMAP (cosine metric, visualization only). "
        "200 classes — colors are not individually distinguishable; "
        "spatial separation indicates cluster structure. "
        "Noise points (×) are excluded from all reported metrics.",
        ha="center", fontsize=7.5, color="#555555", style="italic",
        wrap=True
    )

    plt.tight_layout()
    _save(fig, output_dir, "dbscan_umap_scatter", export_pdf)


def plot_cluster_sizes(labels: np.ndarray,
                       output_dir: str, export_pdf: bool = False,
                       prefix: str = "dbscan") -> None:
    """
    Cluster size distribution: sorted bar + histogram.

    Noise points are excluded (they are not a cluster).
    The true class size (expected ~N/200 for balanced data) is shown
    as a reference line so you can immediately see how spread the
    cluster sizes are relative to expectation.
    """
    valid      = labels[labels >= 0]
    _, counts  = np.unique(valid, return_counts=True)
    counts_s   = np.sort(counts)[::-1]

    expected   = (labels >= 0).sum() / max(1, len(np.unique(valid)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DBSCAN Cluster Size Distribution", fontsize=14)

    # Sorted bar
    axes[0].bar(range(len(counts_s)), counts_s,
                color="#4C72B0", alpha=0.8, width=1.0, edgecolor="none")
    axes[0].axhline(expected, color="#C44E52", linestyle="--", linewidth=1.5,
                    label=f"Mean = {expected:.1f}")
    axes[0].set_xlabel("Cluster (sorted by size)")
    axes[0].set_ylabel("Images per Cluster")
    axes[0].set_title("Cluster Sizes (sorted)")
    axes[0].legend()

    # Histogram
    axes[1].hist(counts_s, bins=40, color="#4C72B0",
                 alpha=0.8, edgecolor="white")
    axes[1].axvline(expected, color="#C44E52", linestyle="--", linewidth=1.5,
                    label=f"Mean = {expected:.1f}")
    axes[1].set_xlabel("Cluster Size")
    axes[1].set_ylabel("Number of Clusters")
    axes[1].set_title("Histogram of Cluster Sizes")
    axes[1].legend()

    fig.text(
        0.5, -0.04,
        "Noise points excluded. A well-calibrated DBSCAN produces roughly "
        "equal cluster sizes for a balanced dataset.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, f"{prefix}_cluster_sizes", export_pdf)


def plot_top_confusions_dbscan(conf_df: pd.DataFrame,
                               output_dir: str, export_pdf: bool = False,
                               prefix: str = "dbscan") -> None:
    """Horizontal bar chart of top confusion pairs after Hungarian mapping."""
    if conf_df is None or conf_df.empty:
        print("  [fig] No confusion pairs (possibly 0 clusters or perfect accuracy).")
        return

    k   = len(conf_df)
    fig, ax = plt.subplots(figsize=(10, max(4, k * 0.38)))

    labels  = [f"{r['true_class']}  →  {r['pred_class']}"
               for _, r in conf_df.iterrows()]
    counts  = conf_df["count"].tolist()
    colors  = plt.cm.Reds(np.linspace(0.35, 0.85, k))[::-1]

    bars = ax.barh(labels, counts, color=colors, edgecolor="none")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", ha="left", fontsize=8)

    ax.set_xlabel("Number of Misclassifications")
    ax.set_title(f"Top {k} Confusion Pairs — DBSCAN  "
                 "(True Class → Predicted Class after Hungarian Mapping)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.text(
        0.5, -0.03,
        "Based on optimal cluster-to-class matching via the Hungarian algorithm. "
        "High count = species most often confused with each other.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, f"{prefix}_top_confusions", export_pdf)


# -------------------------------------------------------
# Main Pipeline Class
# -------------------------------------------------------

class DBSCANPipeline:
    """
    Full DBSCAN workflow:

        prepare_features()      PCA reduction (not UMAP)
        plot_kdistance()        eps guidance
        eps_sweep()             parameter search
        fit_and_evaluate()      final fit with best params
        visualize()             all publication figures

    Usage:
        pipeline = DBSCANPipeline(config)
        results  = pipeline.run(features, true_label_ids, class_names,
                                features_2d, output_dir)
    """

    def __init__(self, config: dict):
        self.config     = config
        self.db_cfg     = config.get("dbscan", {})
        self.export_pdf = config.get("visualization", {}).get("export_pdf", False)

        # PCA dimensionality: 100 is a safe default for 768-D BioCLIP embeddings.
        # It retains ~85-90% of variance while removing noise dimensions.
        self.n_pca      = self.db_cfg.get("pca_components", 100)

        # min_samples for k-distance plot and sweep.
        # Rule of thumb: ln(N) for large datasets, or 2 * D for small ones.
        # For EU-Moth (~4400 images, 200 classes): ~5-10 is sensible.
        self.min_samples = self.db_cfg.get("min_samples", 5)

        # eps sweep range. If not set, it will be auto-computed from the
        # k-distance knee detection.
        self.eps_range  = self.db_cfg.get("eps_range", None)

    def run(self, features: np.ndarray, true_label_ids: np.ndarray,
            class_names: List[str], features_2d: np.ndarray,
            output_dir: str,
            features_reduced: np.ndarray = None,
            reduction_label: str = None) -> dict:
        """
        Run the full pipeline.

        Args:
            features:         Raw L2-normalized embeddings, shape (N, D).
            true_label_ids:   Integer class IDs, shape (N,).
            class_names:      Ordered list of class name strings.
            features_2d:      Pre-computed 2D UMAP projection for visualization.
            output_dir:       Where to save all outputs.
            features_reduced: Pre-reduced features from biocluster.reduction.
                              If provided, skips the internal PCA.
            reduction_label:  Human-readable label for the reduction used
                              (e.g. "UMAP-100D"). Shown in logs.

        Returns:
            Dict with all results and metrics.
        """
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        true_label_ids = np.asarray(true_label_ids, dtype=np.int32)

        print("\n" + "=" * 60)
        print("DBSCAN PIPELINE")
        print("=" * 60)

        # Step 0: Feature space preparation
        if features_reduced is not None:
            label = reduction_label or f"external-{features_reduced.shape[1]}D"
            print(f"\n[Step 0] Using pre-reduced features: {label} "
                  f"({features_reduced.shape[1]}D)")
            features_pca = features_reduced
        else:
            print("\n[Step 0] Feature space preparation (PCA fallback) ...")
            features_pca, pca = prepare_features(features, self.n_pca)

        # Step 1: k-distance plot
        print("\n[Step 1] k-distance plot ...")
        knee_eps = plot_kdistance(
            features_pca, self.min_samples, fig_dir, self.export_pdf
        )

        # Step 2: Determine eps sweep range around the knee
        if self.eps_range is None:
            # Build a range centred on the knee, ±60% of its value
            low  = max(0.01, knee_eps * 0.4)
            high = knee_eps * 1.8
            eps_values = np.linspace(low, high, 20)
            print(f"\n[Step 2] Auto eps range: [{low:.3f}, {high:.3f}] "
                  f"(centred on knee={knee_eps:.3f})")
        else:
            eps_values = np.array(self.eps_range)
            print(f"\n[Step 2] Using config eps range: "
                  f"[{eps_values.min():.3f}, {eps_values.max():.3f}]")

        sweep_df = eps_sweep(
            features_pca, true_label_ids,
            eps_values, self.min_samples,
            fig_dir, self.export_pdf,
            n_true_classes=len(class_names),
        )

        # Step 3: Choose best eps using coverage-weighted ARI
        #
        # We maximise ARI * (1 - noise_pct) rather than ARI alone.
        # Maximising raw ARI selects the smallest eps where a few tiny
        # perfectly-pure clusters form, reporting ARI=1.0 on 3% of data
        # while discarding 97%. The weighted score corrects this by
        # penalising high noise fractions.
        best_row = sweep_df.loc[sweep_df["weighted_score"].idxmax()]
        best_eps = float(best_row["eps"])
        print(f"\n[Step 3] Best eps from sweep: {best_eps:.4f}")
        print(f"  ARI            : {best_row['ari']:.4f}")
        print(f"  Coverage       : {best_row['coverage']:.1%}")
        print(f"  Weighted score : {best_row['weighted_score']:.4f}  (ARI x Coverage)")
        print(f"  Clusters found : {int(best_row['n_clusters'])}")
        print(f"  Noise          : {best_row['noise_pct']:.1%}")

        # Allow manual override from config
        final_eps = self.db_cfg.get("final_eps", best_eps)
        if final_eps != best_eps:
            print(f"  [override] Using config final_eps={final_eps:.4f} "
                  f"instead of auto={best_eps:.4f}")

        # Step 4: Final fit
        print("\n[Step 4] Final DBSCAN fit ...")
        results = fit_and_evaluate(
            features_pca, true_label_ids, class_names,
            eps=final_eps, min_samples=self.min_samples,
            top_k=self.config.get("evaluation", {}).get("top_k_confusions", 20),
        )
        if features_reduced is not None:
            results["reduction"] = reduction_label or "external"
        else:
            results["reduction"] = f"PCA-{self.n_pca}D"
            results["pca_components"] = self.n_pca
        results["knee_eps"] = knee_eps

        # Step 5: Publication figures
        print("\n[Step 5] Generating publication figures ...")

        plot_umap_dbscan(
            features_2d, results["labels"], true_label_ids,
            results, fig_dir, self.export_pdf
        )

        plot_cluster_sizes(results["labels"], fig_dir, self.export_pdf)

        plot_top_confusions_dbscan(
            results.get("conf_df"), fig_dir, self.export_pdf
        )

        # Step 6: Save metrics JSON
        metrics_safe = {
            k: v for k, v in results.items()
            if not isinstance(v, (np.ndarray, pd.DataFrame))
        }
        json_path = os.path.join(output_dir, "dbscan_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_safe, f, indent=2)
        print(f"\n[save] Metrics: {json_path}")

        # Step 7: Save sweep CSV
        csv_path = os.path.join(output_dir, "dbscan_sweep.csv")
        sweep_df.to_csv(csv_path, index=False)
        print(f"[save] Sweep:   {csv_path}")

        print("\n" + "=" * 60)
        print("DBSCAN PIPELINE COMPLETE")
        print(f"  ARI      : {results['ari']:.4f}")
        print(f"  NMI      : {results['nmi']:.4f}")
        print(f"  Accuracy : {results['accuracy']:.4f}")
        print(f"  Clusters : {results['n_clusters']}")
        print(f"  Noise    : {results['n_noise']} ({results['noise_pct']:.1%})")
        print("=" * 60)

        return results
