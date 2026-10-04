"""
Run every clustering method named in the config.

One reduction is applied to all of them. The method is set once, under
reduction.method, and every algorithm receives the same reduced space:

    none      -- the full embedding, 512 to 1280 dimensions by encoder
    pca       -- PCA to n_components dimensions
    umap      -- UMAP to n_components dimensions
    pca+umap  -- PCA to pca_components, then UMAP to n_components

The reduction and dimension used for each reported result are recorded in
configs/paper/, one file per manuscript row.

Usage:
    biocluster-cluster --config configs/paper/eumoth/backbone_bioclip2.yaml

Requires:
    <output_dir>/features.npz    (produced by biocluster-extract)

Output:
    <output_dir>/clusters_<method>.npz        for each method
    <output_dir>/dbscan_metrics.json          if dbscan in config
    <output_dir>/hdbscan_metrics.json         if hdbscan in config
    <output_dir>/density_metrics_combined.json
    <output_dir>/figures/*                    density diagnostic plots
"""

import argparse
import os
import sys
import time

import numpy as np

from biocluster.clustering.methods import run_clustering
from biocluster.reduction import get_reduction_config, reduce_features
from biocluster.utils.io import (
    ensure_dir,
    get_2d_projection,
    load_config,
    load_features,
    save_clusters,
    set_seed,
)
from biocluster.utils.pretty import (
    comparison_table,
    done,
    header,
    metrics_table,
    subheader,
    verdict,
)

# These two route to their own diagnostic pipelines rather than through
# run_clustering, because they choose their own cluster count and can abstain.
# Everything else in methods.py is a plain partition method.
DENSITY_METHODS = {"dbscan", "hdbscan"}


def _run_density(method_name, method_cfg, features, true_label_ids,
                 class_names, features_2d, output_dir, config,
                 features_reduced=None, reduction_label=None):
    """
    Route a density method to its full diagnostic pipeline.

    Args:
        features_reduced: Pre-reduced features (UMAP-100D). If provided,
                          the pipeline skips its internal PCA.
        reduction_label:  Label for the reduction (e.g. "UMAP-100D").
    """
    merged_config = dict(config)
    if method_name == "dbscan":
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("dbscan", {}))
        merged_config["dbscan"] = base
        from biocluster.clustering.density.dbscan_pipeline import DBSCANPipeline
        pipeline = DBSCANPipeline(merged_config)

    elif method_name == "hdbscan":
        base = {k: v for k, v in method_cfg.items() if k != "name"}
        base.update(config.get("hdbscan", {}))
        merged_config["hdbscan"] = base
        from biocluster.clustering.density.hdbscan_pipeline import HDBSCANPipeline
        pipeline = HDBSCANPipeline(merged_config)

    results = pipeline.run(
        features, true_label_ids, class_names, features_2d, output_dir,
        features_reduced=features_reduced,
        reduction_label=reduction_label,
    )

    cluster_path = os.path.join(output_dir, f"clusters_{method_name}.npz")
    save_clusters(cluster_path, results["labels"], method_name, method_cfg)

    # Save HDBSCAN membership probabilities for annotation pipeline
    if method_name == "hdbscan" and results.get("probabilities") is not None:
        probs_path = os.path.join(output_dir, "hdbscan_probabilities.npy")
        np.save(probs_path, results["probabilities"])
        print(f"    [save] HDBSCAN probabilities -> {probs_path}")

    return results


# -------------------------------------------------------
# Final summary table across all methods
# -------------------------------------------------------

def _print_summary(all_results: dict) -> None:
    """Print a unified comparison table after all methods complete."""
    if len(all_results) < 2:
        return

    methods = list(all_results.keys())

    def _get(res, key, default=0.0):
        return res.get(key, default)

    metrics = {
        "ARI":          {m: _get(all_results[m], "ari")      for m in methods},
        "NMI":          {m: _get(all_results[m], "nmi")      for m in methods},
        "Accuracy":     {m: _get(all_results[m], "accuracy") for m in methods},
        "Clusters":     {m: _get(all_results[m], "n_clusters", 0) for m in methods},
        "Noise %":      {m: f"{_get(all_results[m], 'noise_pct', 0.0):.1%}"
                         for m in methods},
    }

    comparison_table(methods, metrics, title="ALL METHODS — FINAL COMPARISON")

    # Identify best method by ARI (density methods use weighted ARI)
    def effective_ari(name):
        r = all_results[name]
        ari = r.get("ari", 0.0)
        if name in DENSITY_METHODS:
            return ari * (1.0 - r.get("noise_pct", 0.0))
        return ari

    best = max(methods, key=effective_ari)
    verdict(best, f"highest ARI={effective_ari(best):.4f}")


def main(config_path: str = None,
         skip_density: bool = False,
         skip_dbscan: bool = False,
         skip_hdbscan: bool = False) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Run all clustering methods (partition + density)."
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config file.")
        args        = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "figures"))

    # Load features
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: features.npz not found at {features_path}")
        print("  Run first:  biocluster-extract  --config <config>")
        sys.exit(1)

    features, labels, image_paths, class_names = load_features(features_path)
    true_label_ids = np.array(labels, dtype=np.int32)

    method_configs = config["clustering"]["methods"]
    method_names   = [m["name"] for m in method_configs]
    has_density    = any(n in DENSITY_METHODS for n in method_names)

    header(f"CLUSTERING  —  {len(method_configs)} method(s)")
    print(f"\n  Dataset : {features.shape[0]} images | "
          f"{len(class_names)} classes | dim={features.shape[1]}")
    print(f"  Methods : {', '.join(method_names)}")

    # ---- Reduction config label ----
    rcfg   = get_reduction_config(config)
    method = rcfg["method"]
    if method == "none":
        reduction_label = f"Full-D ({features.shape[1]})"
    elif method == "pca+umap":
        reduction_label = (f"PCA-{rcfg['pca_components']}D "
                           f"-> UMAP-{rcfg['n_components']}D")
    else:
        reduction_label = f"{method.upper()}-{rcfg['n_components']}D"
    print(f"\n  Reduction : {reduction_label}")

    # ---- Single reduction for all methods ----
    subheader("Dimensionality reduction")
    features_reduced, _ = reduce_features(
        features, config, random_state=config.get("seed", 42)
    )

    # ---- Pre-compute 2D projection for density visualization ----
    features_2d = None
    if has_density:
        subheader("2D projection for density visualization")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="n_jobs value.*overridden",
                category=UserWarning)
            features_2d = get_2d_projection(features, config, output_dir)

    # ---- Run each method ----
    all_results  = {}
    total        = len(method_configs)

    for i, method_cfg in enumerate(method_configs, 1):
        name = method_cfg["name"]
        subheader(f"Method {i}/{total}:  {name.upper()}")
        t0 = time.time()

        print(f"    Reduction: {reduction_label}")

        if name in DENSITY_METHODS:
            # Check skip flags
            if skip_density:
                print("    [skip] density methods (--skip-density)")
                continue
            if name == "dbscan" and skip_dbscan:
                print("    [skip] dbscan (--skip-dbscan)")
                continue
            if name == "hdbscan" and skip_hdbscan:
                print("    [skip] hdbscan (--skip-hdbscan)")
                continue

            results = _run_density(
                name, method_cfg,
                features, true_label_ids, class_names,
                features_2d, output_dir, config,
                features_reduced=features_reduced,
                reduction_label=reduction_label,
            )
            all_results[name] = results

        else:
            cluster_ids = run_clustering(
                method_name=name, features=features_reduced,
                params=method_cfg,
            )
            save_clusters(
                os.path.join(output_dir, f"clusters_{name}.npz"),
                cluster_ids, name, method_cfg,
            )

            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            from biocluster.evaluation.metrics import hungarian_accuracy
            valid = cluster_ids >= 0
            if valid.sum() > 0:
                ari = adjusted_rand_score(true_label_ids[valid], cluster_ids[valid])
                nmi = normalized_mutual_info_score(true_label_ids[valid], cluster_ids[valid])
                acc, _ = hungarian_accuracy(true_label_ids[valid], cluster_ids[valid])
            else:
                ari = nmi = acc = 0.0

            all_results[name] = {
                "ari": ari, "nmi": nmi, "accuracy": acc,
                "n_clusters": len(set(cluster_ids[cluster_ids >= 0])),
                "noise_pct": (cluster_ids < 0).mean(),
            }

            elapsed = time.time() - t0
            metrics_table([
                {"label": "ARI",       "value": ari},
                {"label": "NMI",       "value": nmi},
                {"label": "Accuracy",  "value": acc},
                {"label": "Clusters",  "value": all_results[name]["n_clusters"],
                 "note": f"/ {len(class_names)} expected"},
                {"label": "Reduction", "value": reduction_label},
                {"label": "Time",      "value": f"{elapsed:.1f}s"},
            ], title=f"{name} — quick metrics")

    # Density comparison figure (if both density methods ran)
    d_ran = {n: all_results[n] for n in DENSITY_METHODS if n in all_results}
    if len(d_ran) == 2:
        from biocluster.clustering.density.hdbscan_pipeline import plot_method_comparison
        export_pdf = config.get("visualization", {}).get("export_pdf", False)
        ds         = config.get("dataset", {})
        ds_name = ds.get('name', '')
        ds_classes = ds.get('num_classes', '')
        ds_label = f"{ds_name}, {ds_classes} Species" if ds_name else f"{ds_classes} Species"
        plot_method_comparison(
            d_ran["dbscan"], d_ran["hdbscan"],
            os.path.join(output_dir, "figures"), export_pdf,
            dataset_label=ds_label,
            n_true_classes=len(class_names),
        )

    # ---- Final comparison ----
    _print_summary(all_results)
    done(output_dir)


if __name__ == "__main__":
    main()
