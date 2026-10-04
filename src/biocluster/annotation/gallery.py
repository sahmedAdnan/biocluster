"""
Publication-quality visualization for the annotation pipeline.

Every figure includes:
    - The clustering method name in the title (e.g., "HDBSCAN" or "KMeans")
    - Clear axis labels and interpretation guide
    - Color coding: green = correct/confident, red = wrong/uncertain

Figures generated (all filenames include the method name):
    1. confidence_tiers_{method}.png: How many images are auto/review/human?
    2. signal_breakdown_{method}.png: Which of the 4 signals drives uncertainty?
    3. uncertainty_histogram_{method}.png: Distribution of combined uncertainty
    4. cluster_accuracy_{method}.png: Per-cluster: AI prediction vs ground truth
    5. umap_annotation_{method}.png: Spatial map of correct vs wrong
    6. cluster_grid_{method}.png: Image grid of most uncertain clusters
    7. top_confusions_{method}.png: Side-by-side confused species pairs
    8. provenance_pie_{method}.png: AI vs human (after expert merge only)
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Consistent style
sns.set_style("whitegrid")
COLORS = {
    "auto_accept":    "#55A868",
    "review":         "#CCBB44",
    "human_required": "#CC6677",
    "human_verified": "#4C72B0",
    "correct":        "#55A868",
    "wrong":          "#CC6677",
    "AI_ZeroShot":    "#4C72B0",
    "Human_Expert":   "#DD8452",
}

SIGNAL_NAMES = {
    "semantic_uncertainty":  "Semantic\n(text margin)",
    "spatial_uncertainty":   "Spatial\n(membership)",
    "cluster_agreement":     "Agreement\n(cluster vote)",
    "taxonomic_consistency": "Taxonomic\n(family check)",
}

SIGNAL_EXPLAIN = {
    "semantic_uncertainty":  "many similar-looking species",
    "spatial_uncertainty":   "images on cluster boundaries",
    "cluster_agreement":     "images disagree with cluster majority",
    "taxonomic_consistency": "cluster mixes different families",
}


def _save_fig(fig, output_dir, filename, dpi=300, export_pdf=False):
    """Save figure to PNG (and optionally PDF)."""
    png_path = os.path.join(output_dir, filename)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"    [fig] {png_path}")
    if export_pdf:
        pdf_path = png_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _short(name: str, n: int = 22) -> str:
    """Truncate name for display."""
    name = str(name)
    return name if len(name) <= n else name[:n - 3] + "..."


def _get_viz_cfg(config):
    """Get dpi and export_pdf from config."""
    viz = config.get("visualization", {})
    return viz.get("dpi", 300), viz.get("export_pdf", False)


# -------------------------------------------------------
# 1. Confidence Tier Distribution
# -------------------------------------------------------

def plot_confidence_tiers(df, output_dir, config, method=""):
    """
    Bar chart: how many images fall into each confidence tier.

    Interpretation:
        - auto_accept (green): AI is confident, no review needed.
        - review (yellow): moderate uncertainty, expert should check.
        - human_required (red): high uncertainty, don't trust the AI label.
        - A good clustering method should have mostly green bars.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    tier_order = ["auto_accept", "review", "human_required", "human_verified"]
    tier_counts = df["confidence_tier"].value_counts()
    existing = [t for t in tier_order if t in tier_counts.index]
    counts = [tier_counts[t] for t in existing]
    colors = [COLORS.get(t, "#999999") for t in existing]

    bars = ax.bar(existing, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Confidence Tier", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(f"[{method.upper()}] Confidence Tier Distribution\n"
                 f"green = trust AI, yellow = review, red = needs human",
                 fontsize=13, fontweight="bold")

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"confidence_tiers_{method}.png", dpi, pdf)


def plot_signal_breakdown(df, output_dir, config, method=""):
    """
    Bar chart: mean value of each uncertainty signal across all images.

    Interpretation:
        - The tallest bar is the main source of annotation difficulty.
        - High Semantic = many species look similar (small text margin).
        - High Spatial = images sit on cluster boundaries (HDBSCAN prob low).
        - High Agreement = images within clusters disagree on species.
        - High Taxonomic = cluster mixes species from different families.
        - The weight % shows how much each signal contributes to the
          combined score. Semantic has the highest weight (35%).
    """
    unc_cfg = config.get("annotation", {}).get("uncertainty", {})
    weights = unc_cfg.get("weights", {})
    weight_keys = ["text_margin", "membership_probability",
                   "cluster_agreement", "taxonomic_consistency"]
    signal_cols = list(SIGNAL_NAMES.keys())

    means = [df[col].mean() for col in signal_cols]
    labels = list(SIGNAL_NAMES.values())
    w_values = [weights.get(k, 0) for k in weight_keys]

    # Find the dominant signal
    dominant_idx = np.argmax(means)
    dominant_explain = list(SIGNAL_EXPLAIN.values())[dominant_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#4C72B0", "#DD8452", "#55A868", "#CC6677"]
    bars = ax.bar(labels, means, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, m, w in zip(bars, means, w_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}\n(weight: {w:.0%})",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Mean Uncertainty Score", fontsize=12)
    ax.set_ylim(0, min(1.0, max(means) * 1.5))
    ax.set_title(f"[{method.upper()}] Uncertainty Signal Breakdown\n"
                 f"Dominant signal: {dominant_explain}",
                 fontsize=13, fontweight="bold")

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"signal_breakdown_{method}.png", dpi, pdf)


def plot_uncertainty_histogram(df, output_dir, config, method=""):
    """
    Histogram of combined uncertainty scores with threshold lines.

    Interpretation:
        - The peak shows where most images land on the uncertainty scale.
        - Images LEFT of the orange line are auto-accepted.
        - Images BETWEEN orange and red need expert review.
        - Images RIGHT of the red line require human identification.
        - A narrow peak near 0 = very confident annotations.
        - A wide/bimodal distribution = some clusters are confident,
          others are confused.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = df["combined_uncertainty"].values
    unc_cfg = config.get("annotation", {}).get("uncertainty", {})
    review_thresh = unc_cfg.get("review_threshold", 0.55)
    human_thresh = unc_cfg.get("human_threshold", 0.70)

    sns.histplot(scores, bins=50, kde=True, color=COLORS["AI_ZeroShot"], ax=ax)
    ax.axvline(x=review_thresh, color="orange", linestyle="--", linewidth=2,
               label=f"Review threshold ({review_thresh:.2f})")
    ax.axvline(x=human_thresh, color="red", linestyle="--", linewidth=2,
               label=f"Human threshold ({human_thresh:.2f})")

    ax.set_xlabel("Combined Uncertainty Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"[{method.upper()}] Uncertainty Distribution\n"
                 f"mean={scores.mean():.3f}, median={np.median(scores):.3f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"uncertainty_histogram_{method}.png", dpi, pdf)


def plot_cluster_accuracy(df, output_dir, config, method="", n_show=40):
    """
    Horizontal bar chart: annotation accuracy per cluster, worst first.

    Interpretation:
        - Each row is one cluster. Bar length = % of images correctly labeled.
        - 0% (red, no bar) = AI assigned the WRONG species to the entire cluster.
          The label shows what AI predicted vs what was actually there.
        - 100% (green, full bar) = AI got every image right in this cluster.
        - Partial bars = the cluster contains a mix of species (impure cluster).
        - The 4-signal mini-bar on the right shows WHY this cluster is uncertain:
          S=semantic, P=spatial, A=agreement, T=taxonomic.

    If ground_truth is not available, shows cosine margin (top1 - top2) instead.
    """
    has_gt = "ground_truth" in df.columns

    cluster_data = []
    for cid in df["cluster_id"].unique():
        if cid == -1:
            continue
        subset = df[df["cluster_id"] == cid]
        row = subset.iloc[0]
        entry = {
            "cluster_id": cid,
            "ai_species": row["ai_annotation"],
            "margin": row["margin"],
            "uncertainty": subset["combined_uncertainty"].mean(),
            "size": len(subset),
            "sem": subset["semantic_uncertainty"].mean(),
            "spa": subset["spatial_uncertainty"].mean(),
            "agr": subset["cluster_agreement"].mean(),
            "tax": subset["taxonomic_consistency"].mean(),
        }
        if has_gt:
            gt_counts = subset["ground_truth"].value_counts()
            entry["dominant_gt"] = gt_counts.index[0]
            entry["gt_species_count"] = len(gt_counts)
            correct = (subset["ai_annotation"] == subset["ground_truth"]).sum()
            entry["accuracy"] = correct / len(subset)
            entry["correct"] = correct
        cluster_data.append(entry)

    if not cluster_data:
        return

    if has_gt:
        cluster_data.sort(key=lambda x: x["accuracy"])
    else:
        cluster_data.sort(key=lambda x: x["margin"])
    show = cluster_data[:n_show]

    fig, (ax_bar, ax_signals) = plt.subplots(
        1, 2, figsize=(15, max(6, len(show) * 0.35)),
        gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

    y_pos = range(len(show))

    if has_gt:
        values = [d["accuracy"] for d in show]
        colors = ["#CC6677" if v == 0 else "#CCBB44" if v < 0.8
                  else "#55A868" for v in values]

        # Build labels showing AI prediction vs actual
        labels = []
        for d in show:
            ai = _short(d["ai_species"], 20)
            if d["accuracy"] == 0:
                gt = _short(d["dominant_gt"], 20)
                labels.append(
                    f"C{d['cluster_id']} (n={d['size']}): "
                    f"AI={ai}  ACTUAL={gt}")
            elif d["accuracy"] < 1.0:
                labels.append(
                    f"C{d['cluster_id']} (n={d['size']}): "
                    f"{ai}  ({d['gt_species_count']} species mixed)")
            else:
                labels.append(
                    f"C{d['cluster_id']} (n={d['size']}): {ai}")

        # Draw bars: use minimum width so 0% clusters are visible
        bar_values = [max(v, 0.02) for v in values]
        ax_bar.barh(y_pos, bar_values, color=colors,
                    edgecolor="black", linewidth=0.3)

        # Add accuracy labels
        for i, (v, d) in enumerate(zip(values, show)):
            if v == 0:
                ax_bar.text(0.04, i, "0% - wrong species",
                            va="center", fontsize=8, color="#CC6677",
                            fontweight="bold")
            else:
                ax_bar.text(v + 0.02, i,
                            f"{v:.0%} ({d['correct']}/{d['size']})",
                            va="center", fontsize=8)

        ax_bar.set_xlabel("Annotation Accuracy", fontsize=12)
        ax_bar.set_xlim(0, 1.2)
    else:
        values = [d["margin"] for d in show]
        colors = ["#CC6677" if v < 0.01 else "#CCBB44" if v < 0.03
                  else "#55A868" for v in values]
        labels = [f"C{d['cluster_id']} (n={d['size']}): "
                  f"{_short(d['ai_species'], 22)}" for d in show]
        ax_bar.barh(y_pos, values, color=colors,
                    edgecolor="black", linewidth=0.3)
        ax_bar.set_xlabel("Cosine Margin (top1 - top2)", fontsize=12)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=7.5)
    ax_bar.invert_yaxis()
    ax_bar.set_title(f"[{method.upper()}] Per-Cluster Accuracy "
                     f"(worst {len(show)})", fontsize=13, fontweight="bold")

    # Right panel: 4-signal mini-bars per cluster
    signal_colors = ["#4C72B0", "#DD8452", "#55A868", "#CC6677"]
    bar_width = 0.2
    for i, d in enumerate(show):
        signals = [d["sem"], d["spa"], d["agr"], d["tax"]]
        for j, (s, c) in enumerate(zip(signals, signal_colors)):
            ax_signals.barh(i + (j - 1.5) * bar_width, s,
                            height=bar_width, color=c, alpha=0.8)

    ax_signals.set_xlabel("Signal Value", fontsize=10)
    ax_signals.set_xlim(0, 1.0)
    ax_signals.set_title("Why uncertain?\nS  P  A  T", fontsize=10)

    # Legend for signals
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#4C72B0", label="S: Semantic"),
        Patch(facecolor="#DD8452", label="P: Spatial"),
        Patch(facecolor="#55A868", label="A: Agreement"),
        Patch(facecolor="#CC6677", label="T: Taxonomic"),
    ]
    ax_signals.legend(handles=legend_items, fontsize=7, loc="lower right")

    fig.tight_layout()
    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"cluster_accuracy_{method}.png", dpi, pdf)


def plot_annotation_map(df, features_2d, output_dir, config, method=""):
    """
    UMAP scatter: correct (green) vs wrong (red) annotations spatially.

    Interpretation:
        - Green = AI annotation matches ground truth.
        - Red = AI annotation is wrong.
        - Gray X = noise (not in any cluster).
        - A tight red cluster = clustering grouped wrong species together
          (the cluster itself is bad).
        - Scattered red dots = boundary/overlap (images near cluster edges
          got assigned to the wrong cluster).

    Falls back to confidence tier coloring if no ground_truth.
    """
    has_gt = "ground_truth" in df.columns
    fig, ax = plt.subplots(figsize=(12, 8))

    if has_gt:
        cids = df["cluster_id"].values
        noise = cids == -1
        correct = (~noise) & (df["ai_annotation"].values == df["ground_truth"].values)
        wrong = (~noise) & (~correct)

        if correct.sum() > 0:
            ax.scatter(features_2d[correct, 0], features_2d[correct, 1],
                       c=COLORS["correct"], label=f"Correct ({correct.sum()})",
                       alpha=0.3, s=15, zorder=1)
        if wrong.sum() > 0:
            ax.scatter(features_2d[wrong, 0], features_2d[wrong, 1],
                       c=COLORS["wrong"], label=f"Wrong ({wrong.sum()})",
                       alpha=0.8, s=40, edgecolors="black", linewidth=0.3, zorder=2)
        if noise.sum() > 0:
            ax.scatter(features_2d[noise, 0], features_2d[noise, 1],
                       c="#999999", label=f"Noise ({noise.sum()})",
                       marker="x", alpha=0.5, s=25, zorder=0)

        n_valid = (~noise).sum()
        acc = correct.sum() / n_valid * 100 if n_valid > 0 else 0
        ax.set_title(f"[{method.upper()}] Annotation Accuracy Map\n"
                     f"{acc:.1f}% correct ({correct.sum()}/{n_valid}), "
                     f"{noise.sum()} noise excluded",
                     fontsize=13, fontweight="bold")
    else:
        tiers = df["confidence_tier"].values
        for tier, z, a, s in [("auto_accept", 1, 0.25, 15),
                               ("review", 2, 0.7, 35),
                               ("human_required", 3, 0.9, 60)]:
            mask = tiers == tier
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=COLORS[tier], label=f"{tier} ({mask.sum()})",
                           alpha=a, s=s, zorder=z)
        ax.set_title(f"[{method.upper()}] Confidence Map",
                     fontsize=13, fontweight="bold")

    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.legend(loc="best", fontsize=11, markerscale=1.5)

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"umap_annotation_{method}.png", dpi, pdf)


def plot_cluster_image_grid(df, output_dir, config, method="",
                            n_clusters=15, n_images_per_cluster=5):
    """
    Visual verification: most uncertain clusters as image rows.

    Interpretation:
        - Each ROW is one cluster, sorted by combined uncertainty (worst at top).
        - Left label shows:
            * Cluster ID and size
            * AI prediction and runner-up species
            * Accuracy (% correct in this cluster)
            * 4 signal values: S=semantic P=spatial A=agreement T=taxonomic
        - Each IMAGE overlay:
            * Bottom: "GT: ..." (ground truth) and "AI: ..." (prediction)
            * Top-right: green OK = correct, red X = wrong
        - WHY similar uncertainty gives different results:
            A cluster with unc=0.627 and 100% accuracy means the uncertainty
            comes from signals OTHER than the annotation being wrong (e.g.,
            spatial boundary, taxonomic mixing). A cluster with unc=0.625
            and 0% accuracy means the wrong species was assigned despite
            moderate confidence: the semantic signal was misleading.
            The 4-signal breakdown in the label reveals which signal
            is responsible.
    """
    has_gt = "ground_truth" in df.columns

    cluster_data = []
    for cid in df["cluster_id"].unique():
        if cid == -1:
            continue
        subset = df[df["cluster_id"] == cid]
        row = subset.iloc[0]
        entry = {
            "cluster_id": cid,
            "species": row["ai_annotation"],
            "second_species": row.get("ai_second_choice", ""),
            "margin": row["margin"],
            "uncertainty": subset["combined_uncertainty"].mean(),
            "size": len(subset),
            "indices": subset.index.tolist(),
            "sem": subset["semantic_uncertainty"].mean(),
            "spa": subset["spatial_uncertainty"].mean(),
            "agr": subset["cluster_agreement"].mean(),
            "tax": subset["taxonomic_consistency"].mean(),
        }
        if has_gt:
            correct = (subset["ai_annotation"] == subset["ground_truth"]).sum()
            entry["accuracy"] = correct / len(subset)
        cluster_data.append(entry)

    if not cluster_data:
        return

    cdf = sorted(cluster_data, key=lambda x: -x["uncertainty"])[:n_clusters]

    n_rows = len(cdf)
    n_cols = n_images_per_cluster
    cell_size = 2.8

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_size + 7, n_rows * cell_size),
                             squeeze=False)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    for row_idx, cdata in enumerate(cdf):
        indices = cdata["indices"]
        np.random.seed(42)
        if len(indices) > n_images_per_cluster:
            sel = np.random.choice(len(indices), n_images_per_cluster, replace=False)
            sample_indices = [indices[i] for i in sel]
        else:
            sample_indices = indices

        species = cdata["species"]
        second = cdata["second_species"]

        # Build informative label with signal breakdown
        lines = [f"Cluster {cdata['cluster_id']}  (n={cdata['size']})",
                 f"AI:  {_short(species, 28)}",
                 f"vs:  {_short(second, 26)}"]

        if has_gt:
            acc = cdata["accuracy"]
            lines.append(f"Accuracy: {acc:.0%}")
            row_color = ("#CC6677" if acc < 0.3 else
                         "#CCBB44" if acc < 0.8 else "#55A868")
        else:
            row_color = "#CCBB44"

        # Signal breakdown: show which signal is high
        lines.append(f"Unc: {cdata['uncertainty']:.3f}")
        lines.append(f"  S={cdata['sem']:.2f} P={cdata['spa']:.2f}")
        lines.append(f"  A={cdata['agr']:.2f} T={cdata['tax']:.2f}")

        # Mark the dominant signal
        signal_vals = {"Semantic": cdata["sem"], "Spatial": cdata["spa"],
                       "Agreement": cdata["agr"], "Taxonomic": cdata["tax"]}
        dominant = max(signal_vals, key=signal_vals.get)
        lines.append(f"  >> {dominant}")

        label_text = "\n".join(lines)
        axes[row_idx, 0].set_ylabel(label_text, fontsize=8,
                                     rotation=0, labelpad=145,
                                     ha="right", va="center",
                                     color=row_color, fontweight="bold",
                                     family="monospace")

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(sample_indices):
                idx = sample_indices[col_idx]
                img_path = df.loc[idx, "file_path"]
                gt = df.loc[idx, "ground_truth"] if has_gt else None
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    ax.imshow(np.array(img))
                    _add_image_label(ax, species, ground_truth=gt)
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color="gray")
                    ax.set_facecolor("#f0f0f0")
            else:
                ax.set_facecolor("#f8f8f8")
            for spine in ax.spines.values():
                spine.set_edgecolor(row_color)
                spine.set_linewidth(2.5)

    explanation = (
        "S=Semantic (similar species), P=Spatial (boundary), "
        "A=Agreement (cluster disagrees), T=Taxonomic (family mixing)"
    )
    fig.suptitle(
        f"[{method.upper()}] Most Uncertain Clusters\n{explanation}",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"cluster_grid_{method}.png", dpi, pdf)


def _add_image_label(ax, ai_annotation, ground_truth=None):
    """
    Overlay on image cell: GT vs AI label, and OK/X badge.
    """
    ai_short = _short(ai_annotation, 22)

    if ground_truth is not None:
        gt_short = _short(ground_truth, 22)
        is_match = (ground_truth.lower().replace("_", " ").strip() ==
                    ai_annotation.lower().replace("_", " ").strip())
        label = f"GT: {gt_short}\nAI: {ai_short}"
        badge_color = "#2d8a2d" if is_match else "#cc3333"
        badge_text = "OK" if is_match else "X"
        ax.text(0.96, 0.96, badge_text,
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_color, alpha=0.9))
    else:
        label = f"AI: {ai_short}"

    ax.text(0.5, 0.01, label,
            transform=ax.transAxes, fontsize=7, ha="center", va="bottom",
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75))


def plot_top_confusions(df, output_dir, config, method="",
                        n_confusions=5, n_images=4):
    """
    Side-by-side: the most confused species pairs.

    Interpretation:
        - Each PAIR of rows is one confusion case (smallest margin).
        - Top row (blue): images the AI labeled as species A (its pick).
        - Bottom row (red): images the AI labeled as species B (runner-up).
        - If the two rows look visually similar, the confusion is
          understandable (the species genuinely look alike).
        - If they look different, the clustering mixed unrelated images.
        - The margin value tells how close the decision was (lower = more confused).
    """
    has_gt = "ground_truth" in df.columns

    cluster_data = []
    for cid in df["cluster_id"].unique():
        if cid == -1:
            continue
        subset = df[df["cluster_id"] == cid]
        row = subset.iloc[0]
        cluster_data.append({
            "cluster_id": cid,
            "species_1": row["ai_annotation"],
            "species_2": row.get("ai_second_choice", ""),
            "margin": row["margin"],
            "indices": subset.index.tolist(),
        })

    if not cluster_data:
        return

    confused = sorted(cluster_data, key=lambda x: x["margin"])[:n_confusions]

    # Index for runner-up images
    species_to_indices = {}
    for idx, row in df.iterrows():
        sp = row["ai_annotation"]
        if sp not in species_to_indices:
            species_to_indices[sp] = []
        species_to_indices[sp].append(idx)

    n_pairs = len(confused)
    cell_size = 2.8

    fig, axes = plt.subplots(n_pairs * 2, n_images,
                             figsize=(n_images * cell_size + 5.5,
                                      n_pairs * 2 * cell_size + 2),
                             squeeze=False)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    np.random.seed(42)

    for pair_idx, cdata in enumerate(confused):
        sp1, sp2 = cdata["species_1"], cdata["species_2"]
        margin = cdata["margin"]
        row1, row2 = pair_idx * 2, pair_idx * 2 + 1

        idx1 = cdata["indices"]
        if len(idx1) > n_images:
            sel = np.random.choice(len(idx1), n_images, replace=False)
            idx1 = [idx1[i] for i in sel]

        idx2 = species_to_indices.get(sp2, [])
        if len(idx2) > n_images:
            sel = np.random.choice(len(idx2), n_images, replace=False)
            idx2 = [idx2[i] for i in sel]

        axes[row1, 0].set_ylabel(
            f"AI Pick:\n{_short(sp1, 28)}",
            fontsize=9, rotation=0, labelpad=110,
            ha="right", va="center", color="#4C72B0", fontweight="bold")
        axes[row2, 0].set_ylabel(
            f"Runner-up:\n{_short(sp2, 28)}",
            fontsize=9, rotation=0, labelpad=110,
            ha="right", va="center", color="#CC6677", fontweight="bold")

        axes[row1, n_images // 2].set_title(
            f"Confusion #{pair_idx+1}  -  Cluster {cdata['cluster_id']}  -  "
            f"margin = {margin:.4f}",
            fontsize=11, fontweight="bold", pad=12)

        for col in range(n_images):
            ax = axes[row1, col]
            if col < len(idx1):
                di = idx1[col]
                gt = df.loc[di, "ground_truth"] if has_gt else None
                try:
                    img = Image.open(df.loc[di, "file_path"]).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    ax.imshow(np.array(img))
                    _add_image_label(ax, sp1, ground_truth=gt)
                except Exception:
                    ax.set_facecolor("#f0f0f0")
            else:
                ax.set_facecolor("#f8f8f8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#4C72B0")
                spine.set_linewidth(2.5)

        for col in range(n_images):
            ax = axes[row2, col]
            if col < len(idx2):
                di = idx2[col]
                gt = df.loc[di, "ground_truth"] if has_gt else None
                try:
                    img = Image.open(df.loc[di, "file_path"]).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    ax.imshow(np.array(img))
                    _add_image_label(ax, sp2, ground_truth=gt)
                except Exception:
                    ax.set_facecolor("#f0f0f0")
            else:
                ax.set_facecolor("#f8f8f8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CC6677")
                spine.set_linewidth(2.5)

    fig.suptitle(
        f"[{method.upper()}] Top Confusions - AI Pick vs Runner-Up\n"
        f"(blue = AI's choice, red = what it almost chose, "
        f"margin = cosine gap between the two)",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"top_confusions_{method}.png", dpi, pdf)


# -------------------------------------------------------
# 8. Provenance Pie (after expert merge only)
# -------------------------------------------------------

def plot_provenance_pie(df, output_dir, config, method=""):
    """Pie chart: AI vs Human annotation provenance after expert merge."""
    fig, ax = plt.subplots(figsize=(8, 6))

    source_counts = df["annotation_source"].value_counts()
    colors = [COLORS.get(s, "#999999") for s in source_counts.index]

    ax.pie(source_counts, labels=source_counts.index, autopct="%1.1f%%",
           startangle=140, colors=colors, textprops={"fontsize": 12})
    ax.set_title(f"[{method.upper()}] Annotation Provenance: AI vs Human",
                 fontsize=13, fontweight="bold")

    dpi, pdf = _get_viz_cfg(config)
    _save_fig(fig, output_dir, f"provenance_pie_{method}.png", dpi, pdf)


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------

def generate_gallery(annotations_df: pd.DataFrame,
                     features_2d: Optional[np.ndarray],
                     output_dir: str,
                     config: dict,
                     method_name: str = "unknown") -> None:
    """
    Generate all annotation visualization plots for one clustering method.

    Args:
        annotations_df: Full annotation DataFrame from run_annotation().
        features_2d:    (N, 2) UMAP coordinates for spatial plots.
        output_dir:     Directory to save figures into.
        config:         Full YAML config dict.
        method_name:    Clustering method name (e.g., "hdbscan", "kmeans").
                        Appears in every figure title and filename.
    """
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    has_gt = "ground_truth" in annotations_df.columns
    m = method_name
    print(f"[gallery] Generating figures for {m.upper()} "
          f"({'with' if has_gt else 'without'} ground truth) ...")

    # 1-3: Overview
    plot_confidence_tiers(annotations_df, figures_dir, config, m)
    plot_signal_breakdown(annotations_df, figures_dir, config, m)
    plot_uncertainty_histogram(annotations_df, figures_dir, config, m)

    # 4: Per-cluster detail
    plot_cluster_accuracy(annotations_df, figures_dir, config, m)

    # 5: Spatial map
    if features_2d is not None:
        plot_annotation_map(annotations_df, features_2d, figures_dir, config, m)

    # 6-7: Image grids
    if "file_path" in annotations_df.columns:
        plot_cluster_image_grid(annotations_df, figures_dir, config, m)
        plot_top_confusions(annotations_df, figures_dir, config, m)

    # 8: Provenance (only after expert merge)
    if "Human_Expert" in annotations_df["annotation_source"].values:
        plot_provenance_pie(annotations_df, figures_dir, config, m)

    print(f"[gallery] All {m.upper()} figures saved to {figures_dir}")
