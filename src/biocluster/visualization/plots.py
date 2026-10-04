"""
Publication-ready visualization for BioCLIP-2 clustering experiments.

Every function in this file:
  - Saves a PNG at 300 DPI (+ optional PDF)
  - Has a written caption embedded in the figure explaining how to read it
  - Is documented with a design rationale explaining WHY the visualization
    is designed the way it is, so future users understand the choices
  - Works correctly with 200-class datasets where naive approaches fail

HOW TO READ THE FIGURES (quick reference):
    embedding_true_labels.png   -- where each species lives in 2D space
    embedding_clusters_*.png    -- how each method carved up that space
    confusion_matrix_*.png      -- which species are confused with each other
    top_confusions_*.png        -- the specific pairs causing the most errors
    cluster_sizes_*.png         -- how evenly the clusters were sized
    per_class_accuracy_*.png    -- which individual species had low accuracy
    intra_inter_distances.png   -- how well BioCLIP-2 separated species
    method_comparison.png       -- final scores across all methods
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend; required on headless servers


# -------------------------------------------------------
# Global style
# -------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     100,
    "savefig.bbox":   "tight",
})


# -------------------------------------------------------
# Private helpers
# -------------------------------------------------------

def _save(fig, output_dir, filename, dpi, export_pdf):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print("    [fig]  " + path)
    if export_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                    facecolor="white")


def _subsample(f2d, labels, max_pts):
    if len(labels) <= max_pts:
        return f2d, labels
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), size=max_pts, replace=False)
    return f2d[idx], labels[idx]


def _palette(n):
    if n <= 20:
        return [plt.cm.tab20(i / 20) for i in range(n)]
    return [plt.cm.nipy_spectral(i / n) for i in range(n)]


def _vis(config):
    v = config.get("visualization", {})
    return v.get("dpi", 300), v.get("export_pdf", False), v.get("subsample", 6000)


def plot_embedding(features_2d, labels, title, class_names, config,
                   output_dir, filename, highlight_ids=None):
    """
    Scatter plot of the 2D UMAP or t-SNE projection.

    HOW TO READ THIS FIGURE:
        Each dot is one image. Proximity = similarity in BioCLIP-2 embedding space.
        Same-color dots = same species. Compact, separated clouds = good embedding.
        Mixed, overlapping regions = species the model struggles to distinguish.

    DESIGN FOR 200 CLASSES:
        The eye can only distinguish ~10-12 categorical colors reliably.
        With 200 species, plotting distinct colors for each one is misleading.
        This function uses two layers:

          Grey backdrop:  ALL points as small grey dots -- shows overall structure.
          Colored spotlight: 12 representative classes on top, each with a name
                             in the legend. Default = the 12 largest classes.

        For datasets with 20 or fewer classes, all classes are colored normally.
    """
    dpi, export_pdf, max_pts = _vis(config)
    n_classes = int(labels.max()) + 1
    feat2d, lab2d = _subsample(features_2d, labels, max_pts)

    fig, ax = plt.subplots(figsize=(10, 8))

    if n_classes <= 20:
        pal = _palette(n_classes)
        for cid in range(n_classes):
            mask = lab2d == cid
            if not mask.any():
                continue
            name = (class_names[cid] if cid < len(class_names) else str(cid))
            ax.scatter(feat2d[mask, 0], feat2d[mask, 1],
                       c=[pal[cid]], s=12, alpha=0.70, linewidths=0, label=name)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  frameon=True, fontsize=7, ncol=(2 if n_classes > 10 else 1))
        caption = ("Each color = one class. "
                   "Compact, well-separated clouds indicate good embedding quality.")
    else:
        # Grey backdrop
        ax.scatter(feat2d[:, 0], feat2d[:, 1],
                   c="#CCCCCC", s=5, alpha=0.30, linewidths=0,
                   rasterized=True, zorder=1)
        # Choose spotlight classes
        if highlight_ids is None:
            unique, counts_arr = np.unique(lab2d, return_counts=True)
            highlight_ids = unique[np.argsort(counts_arr)[::-1][:12]].tolist()
        pal = _palette(len(highlight_ids))
        for i, cid in enumerate(highlight_ids):
            mask = lab2d == cid
            if not mask.any():
                continue
            name = (class_names[cid] if cid < len(class_names) else str(cid))
            name = name.replace("_", " ")
            if len(name) > 26:
                name = name[:24] + ".."
            ax.scatter(feat2d[mask, 0], feat2d[mask, 1],
                       c=[pal[i]], s=16, alpha=0.85, linewidths=0,
                       zorder=2, label=name)
        ax.legend(title="Spotlight classes",
                  bbox_to_anchor=(1.02, 1), loc="upper left",
                  frameon=True, fontsize=7, title_fontsize=7)
        caption = (
            "Grey: all " + str(n_classes) + " classes (backdrop = overall structure).  "
            "Colored: " + str(len(highlight_ids)) + " most populated classes (spotlight).  "
            "Clusters in the backdrop indicate species grouped together by the model."
        )

    ax.set_title(title, pad=12, fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.text(0.5, -0.04, caption,
             ha="center", fontsize=7, color="#555555", style="italic", wrap=True)
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


def plot_confusion_heatmap(true_labels, pred_labels, class_names, config,
                           output_dir, filename):
    """
    Two-panel confusion analysis designed for large class counts.

    HOW TO READ THIS FIGURE:
        Left panel -- Per-Class Recall:
            Each bar = one species. Bar length = fraction of that species' images
            correctly assigned (1.0 = perfect). Sorted worst-to-best (worst at top).
            Red = low recall, green = high recall.
            Answers: "Which species is my clustering worst at?"

        Right panel -- Most-Confused Species Submatrix:
            Shows only the K species with the most errors between them.
            Diagonal = correct assignments (want these dark).
            Off-diagonal = confusions (want these light/empty).
            Answers: "Which specific species pairs get mixed up?"

    WHY NOT A FULL 200x200 MATRIX:
        A 200x200 matrix with ~11 images per class produces ~40,000 cells,
        almost all containing 0 or 1. The result is a near-empty grid with
        a faint diagonal -- visually useless. This two-panel design shows
        the same information in a readable way.
    """
    import warnings

    from sklearn.metrics import confusion_matrix

    dpi, export_pdf, _ = _vis(config)
    n  = len(class_names)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n)))

    # Small dataset: full annotated heatmap
    if n <= 30:
        fig, ax = plt.subplots(figsize=(12, 10))
        short = [c[:18].replace("_", " ") for c in class_names]
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=True, fmt="d",
                    annot_kws={"size": 8}, linewidths=0.5,
                    xticklabels=short, yticklabels=short,
                    cbar_kws={"shrink": 0.6, "label": "Count"})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
        ax.set_xlabel("Predicted Class", labelpad=10)
        ax.set_ylabel("True Class",      labelpad=10)
        ax.set_title("Confusion Matrix", pad=14)
        plt.tight_layout()
        _save(fig, output_dir, filename, dpi, export_pdf)
        plt.close(fig)
        return

    # Large dataset: two-panel diagnostic
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Confusion Analysis", fontsize=13, y=1.01)

    # Panel A: per-class recall
    row_sums = cm.sum(axis=1).astype(float)
    row_sums[row_sums == 0] = 1.0
    recall   = cm.diagonal() / row_sums
    show_n   = min(40, n)
    order    = np.argsort(recall)[:show_n]        # worst first
    r_vals   = recall[order]
    r_names  = [class_names[i][:24].replace("_", " ") for i in order]

    bar_colors = plt.cm.RdYlGn(r_vals)
    axes[0].barh(range(show_n), r_vals, color=bar_colors, edgecolor="none")
    axes[0].set_yticks(range(show_n))
    axes[0].set_yticklabels(r_names, fontsize=7)
    axes[0].set_xlabel("Per-Class Recall  (1.0 = all images correct)")
    axes[0].set_title(
        "Per-Class Recall  --  Worst {} Classes\n(sorted: worst at top)".format(show_n),
        fontsize=10
    )
    axes[0].set_xlim(0, 1.08)
    axes[0].invert_yaxis()   # puts index 0 (worst class) at the top
    axes[0].axvline(recall.mean(), color="#333333", linestyle="--", linewidth=1.2,
                    label="Mean = {:.3f}".format(recall.mean()))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)

    # Panel B: most-confused submatrix
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    mass     = off_diag.sum(axis=1) + off_diag.sum(axis=0)
    k_focus  = min(25, n)
    fids     = np.sort(np.argsort(mass)[::-1][:k_focus])
    sub_cm   = cm[np.ix_(fids, fids)]
    sub_names = [class_names[i][:16].replace("_", " ") for i in fids]

    sns.heatmap(sub_cm, ax=axes[1], cmap="Blues",
                annot=(k_focus <= 15), fmt="d",
                annot_kws={"size": 7}, linewidths=0,
                xticklabels=sub_names, yticklabels=sub_names,
                cbar_kws={"shrink": 0.7, "label": "Count"})
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                            rotation=90, ha="right", fontsize=7)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=7)
    axes[1].set_xlabel("Predicted Class", labelpad=8)
    axes[1].set_ylabel("True Class",      labelpad=8)
    axes[1].set_title(
        "Top-{} Most Confused Species\n({}x{} submatrix of full {}x{})".format(
            k_focus, k_focus, k_focus, n, n),
        fontsize=10
    )

    fig.text(
        0.5, -0.02,
        "Left: per-class recall; red = hard species, green = easy species.  "
        "Right: confusion submatrix; diagonal = correct, off-diagonal = errors.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


# -------------------------------------------------------
# 3. Top confusion pairs
# -------------------------------------------------------

def plot_top_confusions(confusion_df, config, output_dir, filename):
    """
    Horizontal bar chart of the most common misclassification pairs.

    HOW TO READ THIS FIGURE:
        Each bar = one pair of species that gets confused.
        Label format: True Species  ->  Predicted Species
        Bar length = number of images where this confusion occurred.
        Top bar = the single most common error in this clustering.
        These are the species pairs worth investigating further --
        they share visual characteristics that confuse the model.
    """
    dpi, export_pdf, _ = _vis(config)

    if confusion_df is None or confusion_df.empty:
        print("    [fig]  skipped top_confusions (no misclassifications found)")
        return

    k      = len(confusion_df)
    labels = [
        str(row["true_class"]) + " -> " + str(row["pred_class"])
        for _, row in confusion_df.iterrows()
    ]
    counts = confusion_df["count"].tolist()
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, k))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, k * 0.38)))
    bars = ax.barh(labels, counts, color=colors, edgecolor="none")

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.15,
                bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", ha="left", fontsize=8)

    ax.set_xlabel("Number of Misclassifications")
    ax.set_title(
        "Top {} Confusion Pairs  (True Species  ->  Predicted Species)".format(k),
        pad=12
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.text(
        0.5, -0.03,
        "Based on optimal cluster-to-species matching (Hungarian algorithm). "
        "High count = species that share visual features confusing to BioCLIP-2.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


# -------------------------------------------------------
# 4. Cluster size distribution
# -------------------------------------------------------

def plot_cluster_sizes(cluster_ids, config, output_dir, filename):
    """
    Two-panel cluster size distribution.

    HOW TO READ THIS FIGURE:
        Left (sorted bar chart):
            Each bar = one cluster, sorted largest to smallest.
            Red dashed line = mean cluster size.
            Ideal: 200 bars of roughly equal height (one per species).
            Very tall bars = one cluster absorbed multiple species.
            Very short bars = a species was fragmented or mostly became noise.

        Right (histogram):
            Distribution of cluster sizes. A narrow peak near (N/K) where
            N = total images and K = expected clusters means well-calibrated.
            Wide/flat distribution = highly uneven cluster sizes.

    Noise points (label = -1, from DBSCAN/HDBSCAN) are excluded.
    """
    dpi, export_pdf, _ = _vis(config)

    valid  = cluster_ids[cluster_ids >= 0]
    _, counts = np.unique(valid, return_counts=True)
    cs  = np.sort(counts)[::-1]
    mu  = float(np.mean(cs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(cs)), cs, color="#4C72B0", alpha=0.8,
                width=1.0, edgecolor="none")
    axes[0].axhline(mu, color="#C44E52", linestyle="--", linewidth=1.5,
                    label="Mean = {:.1f}".format(mu))
    axes[0].set_xlabel("Cluster index (sorted by size)")
    axes[0].set_ylabel("Images in cluster")
    axes[0].set_title("Cluster Sizes (sorted)")
    axes[0].legend()

    axes[1].hist(cs, bins=min(30, len(cs)), color="#4C72B0",
                 alpha=0.8, edgecolor="white")
    axes[1].axvline(mu, color="#C44E52", linestyle="--", linewidth=1.5,
                    label="Mean = {:.1f}".format(mu))
    axes[1].set_xlabel("Cluster size")
    axes[1].set_ylabel("Number of clusters")
    axes[1].set_title("Cluster Size Distribution")
    axes[1].legend()

    fig.suptitle(
        "Cluster Size Analysis  ({} clusters, {} images assigned)".format(
            len(cs), int(valid.shape[0])),
        y=1.02, fontsize=12
    )
    fig.text(
        0.5, -0.04,
        "Noise points excluded. "
        "A well-calibrated clustering produces roughly equal-sized clusters.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


# -------------------------------------------------------
# 5. Intra vs inter-class distance distributions
# -------------------------------------------------------

def plot_intra_inter_distances(features, true_labels, config,
                                output_dir, filename,
                                max_samples_per_class=20):
    """
    Violin + box plot comparing intra-class vs inter-class pairwise distances.

    HOW TO READ THIS FIGURE:
        Blue (Intra-class): distances between images of the SAME species.
        Red  (Inter-class): distances between images of DIFFERENT species.

        Good embedding:
          Blue shifted LEFT  (same species are close together).
          Red shifted RIGHT  (different species are far apart).
          Little overlap between the two distributions.

        Heavy overlap = the embedding does not separate species well.
        No algorithm will cluster reliably if this plot shows heavy overlap.

        Violin shape = full distribution. Inner box = median + IQR.

    Subsampling: distances are O(N^2); we use at most max_samples_per_class
    images per species to keep computation tractable.
    """
    from sklearn.metrics import pairwise_distances

    dpi, export_pdf, _ = _vis(config)
    rng = np.random.default_rng(42)

    sf, sl = [], []
    for cid in np.unique(true_labels):
        idx = np.where(true_labels == cid)[0]
        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)
        sf.append(features[idx])
        sl.extend([cid] * len(idx))

    sf = np.vstack(sf)
    sl = np.array(sl)
    D  = pairwise_distances(sf, metric="euclidean")

    intra, inter = [], []
    n = len(sl)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if sl[i] == sl[j] else inter).append(D[i, j])

    max_per = 20000
    intra = rng.choice(intra, size=min(max_per, len(intra)),
                       replace=False).tolist()
    inter = rng.choice(inter, size=min(max_per, len(inter)),
                       replace=False).tolist()

    df = pd.DataFrame({
        "Distance": intra + inter,
        "Type": ["Intra-class"] * len(intra) + ["Inter-class"] * len(inter),
    })
    pal = {"Intra-class": "#4C72B0", "Inter-class": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    sns.violinplot(data=df, x="Type", y="Distance", hue="Type",
                   palette=pal, inner="box", ax=axes[0],
                   linewidth=1.2, legend=False)
    axes[0].set_title("Distance Distributions (Violin)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Euclidean Distance")

    sns.boxplot(data=df, x="Type", y="Distance", hue="Type",
                palette=pal, width=0.4, linewidth=1.2,
                fliersize=2, ax=axes[1], legend=False)
    axes[1].set_title("Distance Distributions (Box)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Euclidean Distance")

    fig.suptitle("BioCLIP-2 Embedding Separation Quality", y=1.02, fontsize=12)
    fig.text(
        0.5, -0.04,
        "Blue = same species (want left-shifted, narrow).  "
        "Red = different species (want right-shifted, wide).  "
        "More separation = better embedding quality for clustering.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


def plot_method_comparison(metrics_by_method, config, output_dir, filename):
    """
    Bar chart comparing Accuracy, ARI, and NMI across all clustering methods.

    HOW TO READ THIS FIGURE:
        Each group of three bars = one clustering method.
        Blue  = Accuracy: fraction of images in the correct cluster
                          (after optimal cluster-to-species matching).
        Green = ARI:      Adjusted Rand Index, accounts for chance.
        Red   = NMI:      Normalized Mutual Information.

        Higher is better for all three. All values shown as percentages.

        For density methods (DBSCAN, HDBSCAN), scores exclude noise points.
        A method can show high Accuracy but also high noise -- meaning it
        performs well on the images it accepts but rejects many as noise.
    """
    dpi, export_pdf, _ = _vis(config)

    methods  = list(metrics_by_method.keys())
    accuracy = [metrics_by_method[m].get("accuracy", 0) * 100 for m in methods]
    ari      = [metrics_by_method[m].get("ari", 0)      * 100 for m in methods]
    nmi      = [metrics_by_method[m].get("nmi", 0)      * 100 for m in methods]

    x = np.arange(len(methods))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2.2), 6))

    b1 = ax.bar(x - w, accuracy, w, label="Accuracy (%)",
                color="#4C72B0", alpha=0.9)
    b2 = ax.bar(x,     ari,      w, label="ARI (%)",
                color="#55A868", alpha=0.9)
    b3 = ax.bar(x + w, nmi,      w, label="NMI (%)",
                color="#C44E52", alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        "{:.1f}".format(h),
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Clustering Method Comparison", pad=12)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5, -0.04,
        "All scores in percent (0-100). "
        "For density methods, scores are computed on non-noise points only. "
        "Higher = better for all three metrics.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


def plot_per_class_accuracy(per_class_df, config, output_dir, filename):
    """
    Histogram + worst-class bar chart for per-species accuracy.

    HOW TO READ THIS FIGURE:
        Left (histogram):
            Distribution of accuracy across all species. A peak near 1.0
            means most species were correctly clustered. A peak near 0 means
            most species were scattered. The red dashed line = mean accuracy.

        Right (worst classes):
            The 20 species with the lowest individual accuracy.
            Short bar = most images assigned to the wrong cluster.
            These are the species to investigate further in your dataset.
    """
    dpi, export_pdf, _ = _vis(config)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    mean_acc = per_class_df["accuracy"].mean()

    axes[0].hist(per_class_df["accuracy"], bins=20,
                 color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[0].axvline(mean_acc, color="#C44E52", linestyle="--",
                    linewidth=1.5, label="Mean = {:.3f}".format(mean_acc))
    axes[0].set_xlabel("Per-Species Accuracy")
    axes[0].set_ylabel("Number of Species")
    axes[0].set_title("Distribution of Per-Species Accuracy")
    axes[0].legend()

    bottom_n = min(20, len(per_class_df))
    worst    = per_class_df.nsmallest(bottom_n, "accuracy")
    colors   = plt.cm.Reds(np.linspace(0.5, 0.9, bottom_n))[::-1]

    axes[1].barh(
        worst["class_name"].astype(str).str.replace("_", " "),
        worst["accuracy"], color=colors, edgecolor="none"
    )
    axes[1].axvline(mean_acc, color="#C44E52", linestyle="--",
                    linewidth=1.5, label="Mean = {:.3f}".format(mean_acc))
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("Bottom {} Species by Accuracy".format(bottom_n))
    axes[1].set_xlim(0, 1.0)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    fig.suptitle("Per-Species Accuracy Analysis", y=1.02, fontsize=12)
    fig.text(
        0.5, -0.04,
        "Accuracy = fraction of a species' images assigned to its correct cluster "
        "(after optimal Hungarian cluster-to-species mapping).",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)


def plot_cluster_purity(per_cluster_df, config, output_dir, filename):
    """
    Two-panel figure summarising per-cluster purity.

    HOW TO READ THIS FIGURE:
        Left (histogram):
            Distribution of purity scores across all clusters.
            Purity = fraction of images in the cluster that belong to the
            single most common ground-truth species. A value of 1.0 means
            the cluster is perfectly homogeneous (one species only).
            A value near 0 means the cluster mixes many species equally.
            The red dashed line = mean purity across all clusters.

        Right (worst clusters bar chart):
            The 20 clusters with the lowest purity, sorted ascending.
            Each bar shows the purity score; the label shows the cluster ID
            and its size. Low-purity clusters are candidates for further
            investigation: they may indicate species that are visually
            similar enough that the embedding space does not cleanly
            separate them, or that the cluster count K was too low.
    """
    dpi, export_pdf, _ = _vis(config)

    df = per_cluster_df.copy()
    # exclude noise cluster if present
    df = df[df["cluster_id"] >= 0].reset_index(drop=True)

    if df.empty:
        return

    mean_purity = df["purity"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: histogram
    axes[0].hist(df["purity"], bins=20, color="#4C72B0",
                 alpha=0.85, edgecolor="white")
    axes[0].axvline(mean_purity, color="#C44E52", linestyle="--",
                    linewidth=1.5, label="Mean = {:.3f}".format(mean_purity))
    axes[0].set_xlabel("Cluster Purity")
    axes[0].set_ylabel("Number of Clusters")
    axes[0].set_title("Distribution of Per-Cluster Purity")
    axes[0].set_xlim(0, 1.0)
    axes[0].legend()

    # Right: worst clusters bar chart
    bottom_n = min(20, len(df))
    worst = df.nsmallest(bottom_n, "purity").reset_index(drop=True)
    colors = plt.cm.Reds(np.linspace(0.5, 0.9, bottom_n))[::-1]

    labels = [
        "Cluster {} (n={})".format(int(r["cluster_id"]), int(r["size"]))
        for _, r in worst.iterrows()
    ]
    axes[1].barh(labels, worst["purity"], color=colors, edgecolor="none")
    axes[1].axvline(mean_purity, color="#C44E52", linestyle="--",
                    linewidth=1.5, label="Mean = {:.3f}".format(mean_purity))
    axes[1].set_xlabel("Purity")
    axes[1].set_title("Bottom {} Clusters by Purity".format(bottom_n))
    axes[1].set_xlim(0, 1.0)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    fig.suptitle("Per-Cluster Purity Analysis", y=1.02, fontsize=12)
    fig.text(
        0.5, -0.04,
        "Purity = fraction of images in each cluster belonging to the single "
        "most common ground-truth species. A perfectly pure cluster contains "
        "only one species. Low-purity clusters indicate species mixed by the "
        "clustering algorithm.",
        ha="center", fontsize=7, color="#555555", style="italic"
    )
    plt.tight_layout()
    _save(fig, output_dir, filename, dpi, export_pdf)
    plt.close(fig)
