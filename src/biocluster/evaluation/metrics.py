"""
Evaluation metrics for clustering against ground-truth labels.

Computes:
  - ARI  (Adjusted Rand Index):       cluster-label agreement, corrected for chance
  - NMI  (Normalized Mutual Info):    measures shared information
  - Hungarian-mapped accuracy:        optimal cluster-to-class assignment accuracy
  - Per-class accuracy
  - Per-cluster purity
  - Top-K confusion pairs
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)

# -------------------------------------------------------
# Hungarian Algorithm for Cluster-to-Class Mapping
# -------------------------------------------------------

def hungarian_accuracy(true_labels: np.ndarray,
                       cluster_ids: np.ndarray) -> tuple:
    """
    Compute the best possible accuracy by finding the optimal one-to-one
    mapping between cluster IDs and class IDs using the Hungarian algorithm.

    The Hungarian algorithm minimizes cost (or maximizes reward) over all
    possible matchings. Here the reward is the number of correctly mapped
    samples.

    Args:
        true_labels: Ground-truth class IDs, shape (N,).
        cluster_ids: Predicted cluster IDs, shape (N,).

    Returns:
        accuracy:   Float in [0, 1].
        mapping:    Array where mapping[cluster_id] = class_id.
    """
    # Number of unique labels/clusters (use max to handle non-contiguous IDs)
    n = max(cluster_ids.max(), true_labels.max()) + 1

    # Build a co-occurrence (cost) matrix: cost[i, j] = count of samples
    # where cluster=i and class=j
    cost = np.zeros((n, n), dtype=np.int64)
    for c, t in zip(cluster_ids, true_labels):
        cost[c, t] += 1

    # Hungarian algorithm on the negated matrix (maximize matches)
    row_ind, col_ind = linear_sum_assignment(-cost)

    # Total correctly matched samples
    n_correct = cost[row_ind, col_ind].sum()
    accuracy  = n_correct / len(true_labels)

    # Build the full mapping array
    mapping = np.zeros(n, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c

    return accuracy, mapping


# -------------------------------------------------------
# Per-Class and Per-Cluster Metrics
# -------------------------------------------------------

def per_class_accuracy(true_labels: np.ndarray, cluster_ids: np.ndarray,
                       mapping: np.ndarray, class_names: list) -> pd.DataFrame:
    """
    Compute accuracy for each ground-truth class using the Hungarian mapping.

    Args:
        true_labels: Ground-truth class IDs (N,).
        cluster_ids: Cluster assignments (N,).
        mapping:     Cluster-to-class mapping from hungarian_accuracy().
        class_names: List of class name strings.

    Returns:
        DataFrame with columns: class_id, class_name, n_total, n_correct, accuracy.
    """
    # Map cluster predictions to class predictions
    pred_labels = mapping[cluster_ids]

    rows = []
    for class_id, name in enumerate(class_names):
        mask    = true_labels == class_id
        n_total = mask.sum()
        if n_total == 0:
            continue
        n_correct = (pred_labels[mask] == class_id).sum()
        rows.append({
            "class_id":  class_id,
            "class_name": name,
            "n_total":   int(n_total),
            "n_correct": int(n_correct),
            "accuracy":  round(n_correct / n_total, 4),
        })

    return pd.DataFrame(rows).sort_values("accuracy")


def per_cluster_purity(true_labels: np.ndarray,
                       cluster_ids: np.ndarray) -> pd.DataFrame:
    """
    For each cluster, compute purity = (most common true class count) / cluster size.

    A perfectly pure cluster contains only one class.

    Returns:
        DataFrame with columns: cluster_id, size, dominant_class, purity.
    """
    rows = []
    for cid in np.unique(cluster_ids):
        mask          = cluster_ids == cid
        classes_here  = true_labels[mask]
        dominant      = int(np.bincount(classes_here).argmax())
        dominant_cnt  = int(np.bincount(classes_here).max())
        size          = int(mask.sum())
        rows.append({
            "cluster_id":     int(cid),
            "size":           size,
            "dominant_class": dominant,
            "purity":         round(dominant_cnt / size, 4),
        })

    return pd.DataFrame(rows).sort_values("purity")


def top_k_confusion_pairs(true_labels: np.ndarray, pred_labels: np.ndarray,
                           class_names: list, k: int = 20) -> pd.DataFrame:
    """
    Find the K most common misclassification pairs.

    Args:
        true_labels: Ground-truth class IDs.
        pred_labels: Predicted class IDs (after Hungarian mapping).
        class_names: List of class name strings.
        k:           Number of top confusion pairs to return.

    Returns:
        DataFrame with columns: true_class, pred_class, count.
    """
    # Only look at errors
    mask = true_labels != pred_labels
    if mask.sum() == 0:
        return pd.DataFrame(columns=["true_class", "pred_class", "count"])

    true_err = true_labels[mask]
    pred_err = pred_labels[mask]

    # Build confusion counts
    # The warning "number of unique classes > 50% of samples" is a false positive:
    # true_err/pred_err contain only the misclassified subset (small when accuracy
    # is high), but we explicitly pass labels=list(range(n)) so sklearn knows the
    # full class set.
    import warnings
    n = len(class_names)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        cm = confusion_matrix(true_err, pred_err, labels=list(range(n)))

    # Set diagonal to 0 (we only want errors)
    np.fill_diagonal(cm, 0)

    # Get top-K pairs
    flat_indices = np.argsort(cm.ravel())[::-1][:k]
    rows = []
    for idx in flat_indices:
        t = idx // n
        p = idx % n
        cnt = cm[t, p]
        if cnt == 0:
            break
        rows.append({
            "true_class": class_names[t] if t < len(class_names) else str(t),
            "pred_class": class_names[p] if p < len(class_names) else str(p),
            "count":      int(cnt),
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------
# Main Evaluation Function
# -------------------------------------------------------

def evaluate(true_labels: np.ndarray, cluster_ids: np.ndarray,
             class_names: list, top_k: int = 20) -> dict:
    """
    Run the full evaluation suite for one set of cluster assignments.

    Returns a dict with:
      - ari, nmi, accuracy (floats)
      - mapping (list)
      - per_class_df, per_cluster_df, confusion_pairs_df (DataFrames)
      - all metrics also returned as plain floats for JSON serialization
    """
    true_labels = np.asarray(true_labels, dtype=np.int32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)

    # Remove DBSCAN noise points (-1) from evaluation
    valid_mask  = cluster_ids >= 0
    n_noise     = int((~valid_mask).sum())
    if n_noise > 0:
        print(f"  [eval] Excluding {n_noise} noise points (cluster id = -1)")

    t = true_labels[valid_mask]
    c = cluster_ids[valid_mask]

    # Core metrics
    ari = float(adjusted_rand_score(t, c))
    nmi = float(normalized_mutual_info_score(t, c))
    acc, mapping = hungarian_accuracy(t, c)

    # Per-class and per-cluster breakdowns
    per_class_df   = per_class_accuracy(t, c, mapping, class_names)
    per_cluster_df = per_cluster_purity(t, c)

    # Map cluster predictions to class labels for confusion analysis
    pred_labels       = mapping[c]
    confusion_pairs   = top_k_confusion_pairs(t, pred_labels, class_names, k=top_k)

    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}  Hungarian Acc={acc:.4f}")

    return {
        "ari":               ari,
        "nmi":               nmi,
        "accuracy":          float(acc),
        "n_noise_excluded":  n_noise,
        "n_clusters_found":  int(len(np.unique(c))),
        "mapping":           mapping.tolist(),
        "per_class_df":      per_class_df,
        "per_cluster_df":    per_cluster_df,
        "confusion_pairs_df": confusion_pairs,
    }


# -------------------------------------------------------
# Serialization Helpers
# -------------------------------------------------------

def metrics_to_json_safe(metrics: dict) -> dict:
    """
    Strip non-serializable objects (DataFrames) before saving to JSON.
    """
    return {k: v for k, v in metrics.items()
            if not isinstance(v, pd.DataFrame)}


def build_summary_csv(all_metrics: dict, output_path: str) -> None:
    """
    Build and save a summary CSV comparing all methods side by side.

    Args:
        all_metrics: Dict mapping method_name -> metrics dict.
        output_path: Where to save the CSV.
    """
    rows = []
    for method_name, m in all_metrics.items():
        rows.append({
            "method":          method_name,
            "accuracy":        round(m["accuracy"], 4),
            "ari":             round(m["ari"], 4),
            "nmi":             round(m["nmi"], 4),
            "n_clusters_found": m.get("n_clusters_found", "N/A"),
            "n_noise_excluded": m.get("n_noise_excluded", 0),
        })

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"[eval] Summary table saved to {output_path}")
