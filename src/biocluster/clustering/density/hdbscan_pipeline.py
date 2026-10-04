"""
HDBSCAN diagnostic, tuning, and evaluation pipeline.

Feature space:
    By default, the pipeline receives UMAP-100D pre-reduced features from
    the caller.  UMAP preserves local neighborhood topology, aligning with
    HDBSCAN's density-based objective. Validated on EU-Moth (ARI=0.9722)
    and CUB-200 (ARI=0.8745).

    Falls back to internal PCA if no pre-reduced features are provided.

WHY HDBSCAN OVER DBSCAN:
    DBSCAN uses a single global eps threshold. HDBSCAN adapts density
    thresholds per region via a hierarchy over mutual reachability distances.
    For fine-grained biological datasets with variable cluster density,
    HDBSCAN is the more robust choice.

Requirements:
    pip install hdbscan

Usage:
    from biocluster.clustering.density.hdbscan_pipeline import HDBSCANPipeline
    pipeline = HDBSCANPipeline(config)
    results  = pipeline.run(features, true_label_ids, class_names,
                             features_2d, output_dir,
                             features_reduced=umap_features,
                             reduction_label="UMAP-100D")
"""

import json
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

matplotlib.use("Agg")

from biocluster.evaluation.metrics import (
    hungarian_accuracy,
    per_class_accuracy,
    top_k_confusion_pairs,
)

from .dbscan_pipeline import (
    STYLE,
    _save,
    plot_cluster_sizes,
    plot_top_confusions_dbscan,
    prepare_features,
)

def hdbscan_sweep(features: np.ndarray, true_label_ids: np.ndarray,
                  min_cluster_sizes: List[int],
                  output_dir: str, export_pdf: bool = False,
                  n_true_classes: int = None,
                  cluster_selection_method: str = "eom",
                  min_samples: int = None) -> pd.DataFrame:
    """
    Sweep over min_cluster_size (the main HDBSCAN hyperparameter).

    min_cluster_size is analogous to min_samples in DBSCAN: it sets
    the minimum number of points that constitute a cluster. Smaller values
    allow smaller clusters (more clusters, more noise points filtered).
    Larger values force larger clusters (fewer clusters, fewer noise points).

    For EU-Moth with 200 classes and ~22 images/class:
        min_cluster_size = 5-15 is a sensible starting range.
        We sweep from 3 to 30 to cover the full picture.

    Args:
        features:         Reduced feature array (UMAP-100D or PCA).
        true_label_ids:   Integer ground-truth IDs.
        min_cluster_sizes: List of values to try.
        output_dir:       Where to save the sweep plot.
        export_pdf:       Also save PDF.
        min_samples:      Passed straight to HDBSCAN, so the sweep scores the
                          same estimator the final fit uses. None keeps
                          HDBSCAN's own default of min_cluster_size.

    Returns:
        DataFrame with one row per min_cluster_size.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError(
            "HDBSCAN requires: pip install hdbscan\n"
            "On Windows: conda install -c conda-forge hdbscan"
        )

    print(f"  [hdbscan sweep] {len(min_cluster_sizes)} values ...")

    rows = []
    for mcs in min_cluster_sizes:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size         = mcs,
            # Must match the final fit, or Step 2 ranks one estimator and
            # Step 3 fits another. None leaves HDBSCAN's default.
            min_samples              = min_samples,
            metric                   = "euclidean",
            cluster_selection_method = cluster_selection_method,
        )
        labels = clusterer.fit_predict(features)

        n_total    = len(labels)
        n_noise    = int((labels == -1).sum())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = n_noise / n_total

        valid = labels >= 0
        if valid.sum() > 0 and n_clusters > 1:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The number of unique classes",
                                        category=UserWarning)
                ari = adjusted_rand_score(true_label_ids[valid], labels[valid])
                nmi = normalized_mutual_info_score(true_label_ids[valid], labels[valid])
        else:
            ari = nmi = 0.0

        rows.append(dict(
            min_cluster_size=mcs,
            n_clusters=n_clusters,
            noise_pct=noise_pct,
            ari=ari,
            nmi=nmi,
            weighted_ari=ari * (1.0 - noise_pct),
        ))
        print(f"    mcs={mcs:3d} -> clusters={n_clusters:4d}, "
              f"noise={noise_pct:.1%}, ARI={ari:.4f}")

    df = pd.DataFrame(rows)

    # --- Sweep plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HDBSCAN Sweep over min_cluster_size",
                 fontsize=14, y=1.02)

    axes[0].plot(df["min_cluster_size"], df["ari"],
                 color="#4C72B0", linewidth=2, marker="o", markersize=5, label="ARI")
    axes[0].plot(df["min_cluster_size"], df["nmi"],
                 color="#55A868", linewidth=2, marker="s", markersize=5,
                 linestyle="--", label="NMI")
    axes[0].plot(df["min_cluster_size"], df["weighted_ari"],
                 color="#FF7F0E", linewidth=2, marker="^", markersize=5,
                 linestyle=":", label="Weighted ARI (ARI×coverage)")
    axes[0].set_xlabel("min_cluster_size")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Quality vs min_cluster_size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(df["min_cluster_size"], df["noise_pct"] * 100,
                         alpha=0.4, color="#C44E52")
    axes[1].plot(df["min_cluster_size"], df["noise_pct"] * 100,
                 color="#C44E52", linewidth=2, marker="o", markersize=5)
    axes[1].set_xlabel("min_cluster_size")
    axes[1].set_ylabel("% Noise Points")
    axes[1].set_title("Noise Fraction vs min_cluster_size")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["min_cluster_size"], df["n_clusters"],
                 color="#8172B2", linewidth=2, marker="^", markersize=5)
    _true_k = n_true_classes if n_true_classes is not None else 200
    axes[2].axhline(y=_true_k, color="#FF7F0E", linestyle="--", linewidth=1.5,
                    label=f"True # classes ({_true_k})")
    axes[2].set_xlabel("min_cluster_size")
    axes[2].set_ylabel("Clusters Found")
    axes[2].set_title("Clusters Found vs min_cluster_size")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.text(
        0.5, -0.05,
        "Metrics computed on non-noise points only. "
        "HDBSCAN adapts density thresholds per region — expect more stable "
        "cluster counts across the sweep than DBSCAN.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "hdbscan_sweep", export_pdf)

    return df


def hdbscan_fit_and_evaluate(features: np.ndarray,
                              true_label_ids: np.ndarray,
                              class_names: List[str],
                              min_cluster_size: int,
                              min_samples: int = None,
                              top_k: int = 20,
                              cluster_selection_method: str = "eom") -> dict:
    """
    Fit HDBSCAN with chosen parameters and compute all metrics.

    Args:
        features:          Reduced feature array (UMAP-100D or PCA).
        true_label_ids:    Integer class IDs.
        class_names:       Class name strings.
        min_cluster_size:  Main HDBSCAN hyperparameter.
        min_samples:       Controls conservatism of core point selection.
                           If None, defaults to min_cluster_size (HDBSCAN default).
        top_k:             Number of top confusion pairs to report.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError("HDBSCAN requires: pip install hdbscan")

    import time as _time
    print(f"\n  [hdbscan] Final fit: "
          f"min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples or min_cluster_size} ...")
    _t0 = _time.time()

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size         = min_cluster_size,
        min_samples              = min_samples,
        metric                   = "euclidean",
        cluster_selection_method = cluster_selection_method,
    )
    labels = clusterer.fit_predict(features)
    print(f"  [hdbscan] Fit completed in {_time.time() - _t0:.1f}s")

    n_total    = len(labels)
    n_noise    = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"  [hdbscan] Clusters found: {n_clusters}")
    print(f"  [hdbscan] Noise points:   {n_noise} ({n_noise/n_total:.1%})")

    valid   = labels >= 0
    t_valid = true_label_ids[valid]
    c_valid = labels[valid]

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
        mapping    = np.zeros(max(1, labels.max() + 2), dtype=np.int64)
        per_cls_df = pd.DataFrame()
        conf_df    = pd.DataFrame()

    print(f"  [hdbscan] ARI (excl. noise): {ari:.4f}")
    print(f"  [hdbscan] NMI (excl. noise): {nmi:.4f}")
    print(f"  [hdbscan] Acc (excl. noise): {acc:.4f}")

    return {
        "method":           "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples":      min_samples or min_cluster_size,
        "n_clusters":       n_clusters,
        "n_noise":          n_noise,
        "noise_pct":        n_noise / n_total,
        "ari":              ari,
        "nmi":              nmi,
        "accuracy":         float(acc),
        "labels":           labels,
        "mapping":          mapping.tolist(),
        "per_class_df":     per_cls_df,
        "conf_df":          conf_df,
        # HDBSCAN-specific: soft cluster memberships and outlier scores
        "outlier_scores":   getattr(clusterer, "outlier_scores_", None),
        "probabilities":    getattr(clusterer, "probabilities_", None),
    }


# -------------------------------------------------------
# HDBSCAN-specific visualization: outlier score distribution
# -------------------------------------------------------

def plot_outlier_scores(outlier_scores: np.ndarray,
                        output_dir: str, export_pdf: bool = False) -> None:
    """
    Plot the HDBSCAN outlier score distribution.

    Outlier scores range from 0 (well inside a cluster) to 1 (pure outlier).
    A good HDBSCAN result has a bimodal distribution: most points near 0
    (clean cluster members) and a small tail near 1 (true outliers).

    If the distribution is flat or all near 1, the min_cluster_size is
    too small or the embedding density is too uniform for HDBSCAN to work.
    """
    if outlier_scores is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(outlier_scores, bins=60, color="#4C72B0",
            alpha=0.85, edgecolor="white")
    ax.axvline(x=0.5, color="#C44E52", linestyle="--", linewidth=1.5,
               label="Threshold = 0.5")
    high_outlier_pct = (outlier_scores > 0.5).mean() * 100
    ax.set_xlabel("HDBSCAN Outlier Score")
    ax.set_ylabel("Number of Points")
    ax.set_title("HDBSCAN Outlier Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.text(
        0.5, -0.04,
        f"Points with score > 0.5 are likely true outliers "
        f"({high_outlier_pct:.1f}% of data). "
        "A bimodal distribution (peak near 0, tail near 1) indicates "
        "clean cluster separation.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "hdbscan_outlier_scores", export_pdf)


def plot_umap_hdbscan(features_2d: np.ndarray, hdbscan_labels: np.ndarray,
                      true_label_ids: np.ndarray,
                      results: dict,
                      output_dir: str, export_pdf: bool = False,
                      subsample: int = 6000,
                      dataset_label: str = "") -> None:
    """
    Side-by-side scatter: true labels vs HDBSCAN clusters.
    Same layout as the DBSCAN plot for direct visual comparison.
    """
    n   = len(hdbscan_labels)
    rng = np.random.default_rng(42)

    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
    else:
        idx = np.arange(n)

    f2d  = features_2d[idx]
    hlab = hdbscan_labels[idx]
    tlab = true_label_ids[idx]

    n_clusters = results["n_clusters"]
    noise_pct  = results["noise_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title = "HDBSCAN Clustering on BioCLIP-2 Embeddings"
    if dataset_label:
        title += f" ({dataset_label})"
    fig.suptitle(title, fontsize=14, y=1.01)

    # Panel A: True labels
    axes[0].scatter(
        f2d[:, 0], f2d[:, 1],
        c=tlab, cmap=plt.cm.nipy_spectral,
        s=6, alpha=0.5, linewidths=0, rasterized=True
    )
    axes[0].set_title("A)  Ground-Truth Species Labels")
    axes[0].set_xlabel("UMAP Component 1")
    axes[0].set_ylabel("UMAP Component 2")
    axes[0].set_aspect("equal", "datalim")

    # Panel B: HDBSCAN clusters
    noise_mask  = hlab == -1
    valid_mask  = ~noise_mask

    if valid_mask.sum() > 0:
        axes[1].scatter(
            f2d[valid_mask, 0], f2d[valid_mask, 1],
            c=hlab[valid_mask], cmap=plt.cm.nipy_spectral,
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
        f"B)  HDBSCAN Clusters  "
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
        "HDBSCAN adapts density thresholds per region; noise points (×) are "
        "excluded from all reported metrics.",
        ha="center", fontsize=7.5, color="#555555", style="italic",
        wrap=True
    )

    plt.tight_layout()
    _save(fig, output_dir, "hdbscan_umap_scatter", export_pdf)


def plot_method_comparison(dbscan_results: dict, hdbscan_results: dict,
                            output_dir: str, export_pdf: bool = False,
                            dataset_label: str = "",
                            n_true_classes: int = None) -> None:
    """
    Side-by-side bar chart comparing DBSCAN and HDBSCAN on ARI, NMI,
    Accuracy, cluster count, and noise %.

    All metrics exclude noise points (noise is shown separately).
    This is the figure to include in your thesis when discussing the two methods.
    """
    methods     = ["DBSCAN", "HDBSCAN"]
    ari_vals    = [dbscan_results["ari"],      hdbscan_results["ari"]]
    nmi_vals    = [dbscan_results["nmi"],      hdbscan_results["nmi"]]
    acc_vals    = [dbscan_results["accuracy"], hdbscan_results["accuracy"]]
    noise_vals  = [dbscan_results["noise_pct"] * 100,
                   hdbscan_results["noise_pct"] * 100]
    clust_vals  = [dbscan_results["n_clusters"],
                   hdbscan_results["n_clusters"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    title = "DBSCAN vs HDBSCAN — Density Clustering Comparison"
    if dataset_label:
        title += f"  ({dataset_label})"
    fig.suptitle(title, fontsize=13, y=1.02)

    x       = np.arange(len(methods))
    w       = 0.25
    palette = ["#4C72B0", "#55A868", "#C44E52"]

    # Panel 1: ARI, NMI, Accuracy
    bars1 = axes[0].bar(x - w,   ari_vals,  w, color=palette[0],
                        alpha=0.9, label="ARI")
    bars2 = axes[0].bar(x,       nmi_vals,  w, color=palette[1],
                        alpha=0.9, label="NMI")
    bars3 = axes[0].bar(x + w,   acc_vals,  w, color=palette[2],
                        alpha=0.9, label="Accuracy")

    for bgroup in [bars1, bars2, bars3]:
        for bar in bgroup:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Clustering Quality\n(noise excluded)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: # Clusters
    c_bars = axes[1].bar(methods, clust_vals,
                         color=[palette[0], palette[1]], alpha=0.85)
    _true_k = n_true_classes if n_true_classes is not None else 200
    axes[1].axhline(_true_k, color="#FF7F0E", linestyle="--",
                    linewidth=1.5, label=f"True # classes ({_true_k})")
    for bar, val in zip(c_bars, clust_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Number of Clusters Found")
    axes[1].set_title("Clusters Found")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Noise %
    n_bars = axes[2].bar(methods, noise_vals,
                         color=[palette[0], palette[1]], alpha=0.85)
    for bar, val in zip(n_bars, noise_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[2].set_ylabel("% Noise Points")
    axes[2].set_title("Noise Fraction\n(lower is better)")
    axes[2].set_ylim(0, max(noise_vals) * 1.3 + 5)
    axes[2].grid(axis="y", alpha=0.3)

    fig.text(
        0.5, -0.04,
        "All quality metrics (ARI, NMI, Accuracy) are computed on non-noise "
        "points only. Noise % shows the fraction of images that could not be "
        "assigned to any cluster.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "density_method_comparison", export_pdf)


# -------------------------------------------------------
# Main Pipeline Class
# -------------------------------------------------------

class HDBSCANPipeline:
    """
    Full HDBSCAN workflow equivalent to DBSCANPipeline.

    Usage:
        pipeline = HDBSCANPipeline(config)
        results  = pipeline.run(features, true_label_ids, class_names,
                                features_2d, output_dir)
    """

    def __init__(self, config: dict):
        self.config     = config
        self.h_cfg      = config.get("hdbscan", {})
        self.export_pdf = config.get("visualization", {}).get("export_pdf", False)

        self.n_pca = self.h_cfg.get("pca_components", 100)

        # Sweep range for min_cluster_size
        self.sweep_range = self.h_cfg.get(
            "sweep_range",
            list(range(3, 31, 3))   # [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        )

        # "eom" (Excess of Mass, default) merges nested sub-clusters → fewer,
        # larger clusters. Good for compact balanced datasets.
        # "leaf" extracts finest-grained leaf nodes → more, smaller clusters.
        # Recommended for long-tailed datasets with many small classes.
        self.cluster_selection_method = self.h_cfg.get(
            "cluster_selection_method", "eom"
        )

    def run(self, features: np.ndarray, true_label_ids: np.ndarray,
            class_names: List[str], features_2d: np.ndarray,
            output_dir: str,
            features_reduced: np.ndarray = None,
            reduction_label: str = None) -> dict:

        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        true_label_ids = np.asarray(true_label_ids, dtype=np.int32)

        print("\n" + "=" * 60)
        print("HDBSCAN PIPELINE")
        print("=" * 60)

        # Step 0: Feature space preparation
        if features_reduced is not None:
            label = reduction_label or f"external-{features_reduced.shape[1]}D"
            print(f"\n[Step 0] Using pre-reduced features: {label} "
                  f"({features_reduced.shape[1]}D)")
            features_pca = features_reduced
        else:
            print("\n[Step 0] Feature space preparation (PCA fallback) ...")
            features_pca, _ = prepare_features(features, self.n_pca)

        # Step 1: Sweep
        print("\n[Step 1] min_cluster_size sweep ...")
        sweep_df = hdbscan_sweep(
            features_pca, true_label_ids,
            self.sweep_range, fig_dir, self.export_pdf,
            n_true_classes           = len(class_names),
            cluster_selection_method = self.cluster_selection_method,
            min_samples              = self.h_cfg.get("min_samples", None),
        )

        # Step 2: Best min_cluster_size by coverage-weighted ARI.
        # Weighted ARI = ARI × (1 − noise_pct) penalises configurations that
        # discard large fractions of data as noise. Raw ARI alone would favour
        # high-mcs solutions that score well on a small filtered subset while
        # ignoring the rest; this is especially misleading on long-tailed
        # datasets where many rare classes become noise at large mcs.
        best_row = sweep_df.loc[sweep_df["weighted_ari"].idxmax()]
        best_mcs = int(best_row["min_cluster_size"])
        print(f"\n[Step 2] Best min_cluster_size: {best_mcs}  "
              f"(weighted ARI={best_row['weighted_ari']:.4f}, "
              f"raw ARI={best_row['ari']:.4f}, "
              f"noise={best_row['noise_pct']:.1%})")

        final_mcs = self.h_cfg.get("final_min_cluster_size", best_mcs)
        if final_mcs != best_mcs:
            print(f"  [override] Using config value={final_mcs}")

        # Step 3: Final fit
        print("\n[Step 3] Final HDBSCAN fit ...")
        results = hdbscan_fit_and_evaluate(
            features_pca, true_label_ids, class_names,
            min_cluster_size         = final_mcs,
            min_samples              = self.h_cfg.get("min_samples", None),
            top_k                    = self.config.get("evaluation", {}).get("top_k_confusions", 20),
            cluster_selection_method = self.cluster_selection_method,
        )
        if features_reduced is not None:
            results["reduction"] = reduction_label or "external"
        else:
            results["reduction"] = f"PCA-{self.n_pca}D"
            results["pca_components"] = self.n_pca

        # Step 4: Figures
        print("\n[Step 4] Generating publication figures ...")

        ds       = self.config.get("dataset", {})
        ds_name = ds.get('name', '')
        ds_classes = ds.get('num_classes', '')
        ds_label = f"{ds_name}, {ds_classes} Species" if ds_name else f"{ds_classes} Species"
        plot_umap_hdbscan(
            features_2d, results["labels"], true_label_ids,
            results, fig_dir, self.export_pdf, dataset_label=ds_label
        )

        plot_cluster_sizes(results["labels"], fig_dir, self.export_pdf,
                           prefix="hdbscan")

        if results.get("outlier_scores") is not None:
            plot_outlier_scores(
                results["outlier_scores"], fig_dir, self.export_pdf
            )

        plot_top_confusions_dbscan(
            results.get("conf_df"), fig_dir, self.export_pdf,
            prefix="hdbscan"
        )

        # Step 5: Save metrics
        metrics_safe = {
            k: v for k, v in results.items()
            if not isinstance(v, (np.ndarray, pd.DataFrame))
        }
        # Convert outlier_scores list if present
        if "outlier_scores" in metrics_safe and metrics_safe["outlier_scores"] is not None:
            metrics_safe["outlier_scores"] = None   # too large for JSON

        json_path = os.path.join(output_dir, "hdbscan_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_safe, f, indent=2)
        print(f"\n[save] Metrics: {json_path}")

        sweep_df.to_csv(
            os.path.join(output_dir, "hdbscan_sweep.csv"), index=False
        )

        print("\n" + "=" * 60)
        print("HDBSCAN PIPELINE COMPLETE")
        print(f"  ARI      : {results['ari']:.4f}")
        print(f"  NMI      : {results['nmi']:.4f}")
        print(f"  Accuracy : {results['accuracy']:.4f}")
        print(f"  Clusters : {results['n_clusters']}")
        print(f"  Noise    : {results['n_noise']} ({results['noise_pct']:.1%})")
        print("=" * 60)

        return results
