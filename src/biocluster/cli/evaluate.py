"""
CLI command: evaluate all clustering results and generate all figures.

Usage:
    biocluster-evaluate --config configs/paper/eumoth/backbone_bioclip2.yaml

Requires:
    <output_dir>/features.npz
    <output_dir>/clusters_<method>.npz   for each method

Output:
    <output_dir>/metrics.json
    <output_dir>/metrics_summary.csv
    <output_dir>/figures/*.png
"""

import argparse
import json
import os
import time

import numpy as np

from biocluster.evaluation.metrics import build_summary_csv, evaluate, metrics_to_json_safe

from biocluster.utils.io import (
    ensure_dir,
    get_2d_projection,
    load_clusters,
    load_config,
    load_features,
    set_seed,
)
from biocluster.utils.pretty import (
    comparison_table,
    done,
    header,
    metrics_table,
    saved,
    subheader,
    verdict,
)
from biocluster.visualization.plots import (
    plot_cluster_purity,
    plot_cluster_sizes,
    plot_confusion_heatmap,
    plot_embedding,
    plot_intra_inter_distances,
    plot_method_comparison,
    plot_per_class_accuracy,
    plot_top_confusions,
)


def main(config_path: str = None) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Evaluate clustering results and generate figures."
        )
        parser.add_argument("--config", required=True)
        args        = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir  = config["output_dir"]
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)

    header("STEP: Evaluation + Figures")

    # Load features
    fp = os.path.join(output_dir, "features.npz")
    features, labels, image_paths, class_names = load_features(fp)
    true_labels = np.array(labels)
    top_k = config["evaluation"].get("top_k_confusions", 20)

    print("    Images    : " + str(len(true_labels)))
    print("    Classes   : " + str(len(class_names)))

    # 2D projection (shared across all plots)
    subheader("2D projection")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message="n_jobs value.*overridden",
            category=UserWarning)
        features_2d = get_2d_projection(features, config, output_dir)

    # True-label embedding (shared reference figure)
    subheader("True-label embedding")
    print("    Generating true-label embedding scatter ...")
    plot_embedding(
        features_2d, true_labels,
        title       = "2D Embedding  --  True Labels  (" + config["dataset"]["name"] + ")",
        class_names = class_names,
        config      = config,
        output_dir  = figures_dir,
        filename    = "embedding_true_labels.png",
    )

    # Intra/inter distances (computed once)
    subheader("Embedding separation quality")
    print("    Computing intra/inter class distances ...")
    plot_intra_inter_distances(
        features, true_labels, config,
        output_dir = figures_dir,
        filename   = "intra_inter_distances.png",
    )

    # Evaluate each method
    all_metrics    = {}
    method_configs = {m["name"]: m for m in config["clustering"]["methods"]}
    total          = len(method_configs)

    for i, (method_name, method_cfg) in enumerate(method_configs.items(), 1):
        cluster_path = os.path.join(output_dir, "clusters_" + method_name + ".npz")
        if not os.path.exists(cluster_path):
            print("\n    [skip]  " + method_name + " (no cluster file found)")
            continue

        subheader("Method {}/{}:  {}".format(i, total, method_name.upper()))
        t0 = time.time()

        cluster_ids, _, _ = load_clusters(cluster_path)
        metrics = evaluate(true_labels, cluster_ids, class_names, top_k=top_k)
        all_metrics[method_name] = metrics

        n_noise    = int((cluster_ids < 0).sum())
        noise_pct  = n_noise / len(cluster_ids)

        metrics_table([
            {"label": "ARI",       "value": metrics["ari"]},
            {"label": "NMI",       "value": metrics["nmi"]},
            {"label": "Accuracy",  "value": metrics["accuracy"]},
            {"label": "Clusters",  "value": len(set(cluster_ids[cluster_ids >= 0])),
             "note": "/ " + str(len(class_names)) + " expected"},
            {"label": "Noise",     "value": "{:.1%}".format(noise_pct)},
            {"label": "Time",      "value": "{:.1f}s".format(time.time() - t0)},
        ], title=method_name + "  results")

        # Figures for this method
        print(f"    Generating figures for {method_name} ...")
        valid = cluster_ids >= 0
        plot_embedding(
            features_2d[valid], cluster_ids[valid],
            title       = "2D Embedding  --  " + method_name + " Clusters",
            class_names = [str(i) for i in range(int(cluster_ids[valid].max()) + 1)],
            config      = config,
            output_dir  = figures_dir,
            filename    = "embedding_clusters_" + method_name + ".png",
        )

        mapping     = np.array(metrics["mapping"])
        pred_labels = mapping[cluster_ids[valid]]
        plot_confusion_heatmap(
            true_labels[valid], pred_labels,
            class_names = class_names,
            config      = config,
            output_dir  = figures_dir,
            filename    = "confusion_matrix_" + method_name + ".png",
        )
        plot_top_confusions(
            metrics["confusion_pairs_df"], config,
            output_dir = figures_dir,
            filename   = "top_confusions_" + method_name + ".png",
        )
        plot_cluster_sizes(
            cluster_ids, config,
            output_dir = figures_dir,
            filename   = "cluster_sizes_" + method_name + ".png",
        )
        plot_per_class_accuracy(
            metrics["per_class_df"], config,
            output_dir = figures_dir,
            filename   = "per_class_accuracy_" + method_name + ".png",
        )

        # Save per-cluster purity CSV
        purity_csv = os.path.join(output_dir,
                                  "per_cluster_purity_" + method_name + ".csv")
        metrics["per_cluster_df"].to_csv(purity_csv, index=False)
        saved("per_cluster_purity_" + method_name + ".csv", purity_csv)

        # Per-cluster purity figure
        plot_cluster_purity(
            metrics["per_cluster_df"], config,
            output_dir = figures_dir,
            filename   = "per_cluster_purity_" + method_name + ".png",
        )

    # Cross-method comparison
    if len(all_metrics) > 1:
        subheader("Method comparison figure")
        print("    Generating cross-method comparison ...")
        plot_method_comparison(
            all_metrics, config,
            output_dir = figures_dir,
            filename   = "method_comparison.png",
        )

        # CLI comparison table
        methods = list(all_metrics.keys())
        comparison_table(
            methods=methods,
            metrics={
                "ARI":      {m: all_metrics[m]["ari"]      for m in methods},
                "NMI":      {m: all_metrics[m]["nmi"]      for m in methods},
                "Accuracy": {m: all_metrics[m]["accuracy"] for m in methods},
            },
            title="All Methods  --  Final Scores"
        )
        best = max(methods, key=lambda m: all_metrics[m]["ari"])
        verdict(best, "highest ARI = {:.4f}".format(all_metrics[best]["ari"]))

    # Save metrics
    subheader("Saving metrics")
    metrics_json = {n: metrics_to_json_safe(m) for n, m in all_metrics.items()}
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    saved("metrics.json", json_path)

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    build_summary_csv(all_metrics, csv_path)
    saved("metrics_summary.csv", csv_path)

    done(output_dir)


if __name__ == "__main__":
    main()
