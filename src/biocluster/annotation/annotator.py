"""
Core annotation engine: assign species names to clusters via zero-shot
text matching in 768D CLIP space, then quantify uncertainty with 4 signals.

Pipeline:
    1. Compute cluster centroids in original 768D embedding space
    2. Cosine-match each centroid to species text embeddings
    3. Per-image voting: each image votes for its nearest text embedding
    4. Compute 4 uncertainty signals per image
    5. Classify each image: auto_accept / review / human_required

Uncertainty signals:
    1. Semantic  (text_margin):        1 - (top1_cosine - top2_cosine)
    2. Spatial   (membership_prob):    1 - HDBSCAN_probability (or silhouette)
    3. Cluster agreement:              1 - (fraction of cluster voting for same species)
    4. Taxonomic consistency:          1 if cluster mixes families, 0 if pure
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Cluster Centroid Annotation
# -------------------------------------------------------

def annotate_clusters(features: np.ndarray,
                      cluster_ids: np.ndarray,
                      text_embs: np.ndarray,
                      species_names: List[str],
                      ) -> Dict[int, dict]:
    """
    Annotate each cluster by matching its centroid to species text embeddings.

    Args:
        features:      (N, D) image embeddings in original 768D space.
        cluster_ids:   (N,) integer cluster assignments (-1 = noise).
        text_embs:     (S, D) species text embeddings from encode_text().
        species_names: (S,) list of species name strings.

    Returns:
        Dict mapping cluster_id -> {
            "species": str,
            "confidence": float (cosine similarity),
            "second_species": str,
            "second_confidence": float,
            "margin": float (top1 - top2),
            "size": int,
        }
    """
    unique_clusters = np.unique(cluster_ids)
    annotations = {}

    for cid in unique_clusters:
        if cid == -1:
            annotations[cid] = {
                "species": "Noise / Unknown",
                "confidence": 0.0,
                "second_species": "",
                "second_confidence": 0.0,
                "margin": 0.0,
                "size": int((cluster_ids == cid).sum()),
            }
            continue

        # Compute centroid in 768D
        mask = cluster_ids == cid
        centroid = features[mask].mean(axis=0).reshape(1, -1)

        # Cosine similarity to all species
        sims = cosine_similarity(centroid, text_embs)[0]
        top2_idx = np.argsort(sims)[-2:][::-1]

        annotations[cid] = {
            "species":             species_names[top2_idx[0]],
            "confidence":          float(sims[top2_idx[0]]),
            "second_species":      species_names[top2_idx[1]],
            "second_confidence":   float(sims[top2_idx[1]]),
            "margin":              float(sims[top2_idx[0]] - sims[top2_idx[1]]),
            "size":                int(mask.sum()),
        }

    return annotations


# -------------------------------------------------------
# Per-Image Voting (for cluster agreement signal)
# -------------------------------------------------------

def compute_image_votes(features: np.ndarray,
                        cluster_ids: np.ndarray,
                        text_embs: np.ndarray,
                        species_names: List[str],
                        ) -> Tuple[List[str], List[str], np.ndarray]:
    """
    For each image, find its nearest species text embedding independently.

    Returns:
        image_top1:  (N,) list of top-1 species per image
        image_top2:  (N,) list of top-2 species per image
        image_sims:  (N, S) full similarity matrix
    """
    # Batch cosine similarity: (N, S)
    sims = cosine_similarity(features, text_embs)

    top2_indices = np.argsort(sims, axis=1)[:, -2:][:, ::-1]

    image_top1 = [species_names[top2_indices[i, 0]] for i in range(len(features))]
    image_top2 = [species_names[top2_indices[i, 1]] for i in range(len(features))]

    return image_top1, image_top2, sims


# -------------------------------------------------------
# Uncertainty Signal 1: Semantic (text margin)
# -------------------------------------------------------

def compute_semantic_uncertainty(image_sims: np.ndarray) -> np.ndarray:
    """
    Semantic uncertainty = 1 - margin between top-1 and top-2 cosine scores.

    A small margin means the model can't clearly distinguish between two species.

    Args:
        image_sims: (N, S) cosine similarity matrix.

    Returns:
        (N,) array of semantic uncertainty scores in [0, 1].
    """
    top2 = np.sort(image_sims, axis=1)[:, -2:]  # (N, 2)
    margin = top2[:, 1] - top2[:, 0]            # top1 - top2
    return 1.0 - margin


# -------------------------------------------------------
# Uncertainty Signal 2: Spatial (HDBSCAN probability)
# -------------------------------------------------------

def compute_spatial_uncertainty(cluster_ids: np.ndarray,
                                probabilities: Optional[np.ndarray] = None,
                                features: Optional[np.ndarray] = None,
                                ) -> np.ndarray:
    """
    Spatial uncertainty from HDBSCAN membership probability.

    If HDBSCAN probabilities are available:
        uncertainty = 1 - probability  (noise points get 1.0)

    Fallback (non-HDBSCAN methods): per-sample silhouette score.
        uncertainty = (1 - silhouette) / 2  (scaled to [0, 1])

    Args:
        cluster_ids:   (N,) cluster assignments.
        probabilities: (N,) HDBSCAN membership probabilities (optional).
        features:      (N, D) features for silhouette fallback (optional).

    Returns:
        (N,) array of spatial uncertainty scores in [0, 1].
    """
    n = len(cluster_ids)

    if probabilities is not None:
        uncertainty = np.ones(n)
        valid = cluster_ids >= 0
        uncertainty[valid] = 1.0 - probabilities[valid]
        return uncertainty

    # Fallback: silhouette score
    if features is not None:
        valid = cluster_ids >= 0
        n_valid_clusters = len(set(cluster_ids[valid]))

        if n_valid_clusters >= 2:
            from sklearn.metrics import silhouette_samples
            uncertainty = np.ones(n)
            sil = silhouette_samples(features[valid], cluster_ids[valid],
                                     metric="cosine")
            # silhouette in [-1, 1], map to [0, 1]: uncertainty = (1 - sil) / 2
            uncertainty[valid] = (1.0 - sil) / 2.0
            return uncertainty

    # No spatial signal available
    return np.full(n, 0.5)


def compute_cluster_agreement(cluster_ids: np.ndarray,
                              image_top1: List[str],
                              ) -> np.ndarray:
    """
    Cluster agreement: what fraction of images in each cluster vote
    for the same species as the majority?

    Per-image uncertainty = 1 - (majority_fraction in this image's cluster)

    If an image votes differently from the cluster majority, it gets
    additional uncertainty from being a dissenter.

    Args:
        cluster_ids: (N,) cluster assignments.
        image_top1:  (N,) per-image top-1 species vote.

    Returns:
        (N,) array of cluster agreement uncertainty in [0, 1].
    """
    n = len(cluster_ids)
    uncertainty = np.zeros(n)

    # Group images by cluster
    cluster_to_indices = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(i)

    for cid, indices in cluster_to_indices.items():
        if cid == -1:
            # Noise points: maximum agreement uncertainty
            for i in indices:
                uncertainty[i] = 1.0
            continue

        # Count votes within this cluster
        votes = [image_top1[i] for i in indices]
        counter = Counter(votes)
        majority_species, majority_count = counter.most_common(1)[0]
        majority_fraction = majority_count / len(indices)

        for i in indices:
            # Base uncertainty from cluster disagreement
            base = 1.0 - majority_fraction

            # Extra penalty if this image dissents from majority
            if image_top1[i] != majority_species:
                base = min(1.0, base + 0.2)

            uncertainty[i] = base

    return uncertainty


def compute_taxonomic_consistency(cluster_ids: np.ndarray,
                                  image_top1: List[str],
                                  taxonomy_table: dict,
                                  ) -> np.ndarray:
    """
    Taxonomic consistency: does the cluster mix species from different families?

    If all image-level votes within a cluster share the same family -> 0.0 (pure).
    If votes span multiple families -> scaled by how many families are mixed.

    Args:
        cluster_ids:    (N,) cluster assignments.
        image_top1:     (N,) per-image top-1 species vote.
        taxonomy_table: Dict mapping species_name -> {"family": str, ...}.

    Returns:
        (N,) array of taxonomic inconsistency in [0, 1].
    """
    n = len(cluster_ids)
    uncertainty = np.zeros(n)

    if not taxonomy_table:
        return uncertainty

    # Group by cluster
    cluster_to_indices = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(i)

    for cid, indices in cluster_to_indices.items():
        if cid == -1:
            for i in indices:
                uncertainty[i] = 1.0
            continue

        # Collect families of voted species
        families = set()
        for i in indices:
            species = image_top1[i]
            # Look up family: try both original and readable name
            tax = taxonomy_table.get(species, {})
            if not tax:
                # Try with underscores replaced
                tax = taxonomy_table.get(species.replace(" ", "_"), {})
            family = tax.get("family", "")
            if family:
                families.add(family)

        # If we couldn't look up any families, no signal
        if len(families) <= 1:
            # Pure or unknown: no inconsistency
            for i in indices:
                uncertainty[i] = 0.0
        else:
            # Mixed families: uncertainty proportional to diversity
            # 2 families -> 0.5, 3+ families -> higher
            inconsistency = min(1.0, (len(families) - 1) / 3.0)
            for i in indices:
                uncertainty[i] = inconsistency

    return uncertainty


def compute_combined_uncertainty(
        semantic: np.ndarray,
        spatial: np.ndarray,
        agreement: np.ndarray,
        taxonomic: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Weighted average of all 4 uncertainty signals.

    Default weights: semantic=0.35, spatial=0.25, agreement=0.25, taxonomic=0.15

    Returns:
        (N,) combined uncertainty score in [0, 1].
    """
    if weights is None:
        weights = {
            "text_margin": 0.35,
            "membership_probability": 0.25,
            "cluster_agreement": 0.25,
            "taxonomic_consistency": 0.15,
        }

    w_sem = weights.get("text_margin", 0.35)
    w_spa = weights.get("membership_probability", 0.25)
    w_agr = weights.get("cluster_agreement", 0.25)
    w_tax = weights.get("taxonomic_consistency", 0.15)

    total_w = w_sem + w_spa + w_agr + w_tax
    combined = (w_sem * semantic + w_spa * spatial +
                w_agr * agreement + w_tax * taxonomic) / total_w

    return combined


def classify_confidence(combined_uncertainty: np.ndarray,
                        config: dict,
                        ) -> List[str]:
    """
    Classify each image into confidence tiers based on combined uncertainty.

    Uses absolute thresholds on the combined uncertainty score:
        - combined < review_threshold  -> auto_accept
        - review_threshold <= combined < human_threshold -> review
        - combined >= human_threshold  -> human_required

    The thresholds are configurable via:
        annotation.uncertainty.review_threshold   (default: 0.65)
        annotation.uncertainty.human_threshold    (default: 0.80)

    Why absolute thresholds instead of percentile-based:
        Percentile thresholds (e.g., "flag top 5%") guarantee a fixed fraction
        of images are flagged regardless of actual uncertainty. This means a
        method with universally high confidence and a method with widespread
        confusion both flag exactly the same percentage, hiding real differences.
        Absolute thresholds let the data speak: a good clustering method can
        have 0 flagged images, while a poor one may flag hundreds.

    Returns:
        (N,) list of "auto_accept", "review", or "human_required".
    """
    ann_cfg = config.get("annotation", {})
    unc_cfg = ann_cfg.get("uncertainty", {})

    review_threshold = unc_cfg.get("review_threshold", 0.65)
    human_threshold  = unc_cfg.get("human_threshold", 0.80)

    classifications = []
    for u in combined_uncertainty:
        if u >= human_threshold:
            classifications.append("human_required")
        elif u >= review_threshold:
            classifications.append("review")
        else:
            classifications.append("auto_accept")

    return classifications


def run_annotation(features: np.ndarray,
                   cluster_ids: np.ndarray,
                   text_embs: np.ndarray,
                   species_names: List[str],
                   taxonomy_table: dict,
                   config: dict,
                   probabilities: Optional[np.ndarray] = None,
                   image_paths: Optional[List[str]] = None,
                   ) -> pd.DataFrame:
    """
    Full annotation pipeline: annotate clusters, compute all 4 uncertainty
    signals, classify confidence, and return results as a DataFrame.

    Args:
        features:       (N, 768) image embeddings in original CLIP space.
        cluster_ids:    (N,) cluster assignments from any method.
        text_embs:      (S, D) species text embeddings.
        species_names:  (S,) species name strings.
        taxonomy_table: Dict from build_taxonomy_table().
        config:         Full YAML config dict.
        probabilities:  (N,) HDBSCAN membership probabilities (optional).
        image_paths:    (N,) image file paths (optional, for output).

    Returns:
        DataFrame with columns:
            file_path, cluster_id, ai_annotation, ai_second_choice,
            confidence, margin, semantic_uncertainty, spatial_uncertainty,
            cluster_agreement, taxonomic_consistency, combined_uncertainty,
            confidence_tier, annotation_source
    """
    n = len(features)

    # Step 1: Annotate clusters (centroid -> text matching)
    print("[annotate] Step 1/4: Cluster centroid annotation ...")
    cluster_annotations = annotate_clusters(
        features, cluster_ids, text_embs, species_names
    )

    n_clusters = len([c for c in cluster_annotations if c != -1])
    print(f"    Annotated {n_clusters} clusters")

    print("[annotate] Step 2/4: Per-image text matching ...")
    image_top1, image_top2, image_sims = compute_image_votes(
        features, cluster_ids, text_embs, species_names
    )

    print("[annotate] Step 3/4: Computing 4 uncertainty signals ...")

    semantic = compute_semantic_uncertainty(image_sims)
    print(f"    Semantic  (text margin):  mean={semantic.mean():.3f}")

    spatial = compute_spatial_uncertainty(cluster_ids, probabilities, features)
    print(f"    Spatial   (membership):   mean={spatial.mean():.3f}")

    agreement = compute_cluster_agreement(cluster_ids, image_top1)
    print(f"    Agreement (cluster vote): mean={agreement.mean():.3f}")

    taxonomic = compute_taxonomic_consistency(
        cluster_ids, image_top1, taxonomy_table
    )
    print(f"    Taxonomic (family check): mean={taxonomic.mean():.3f}")

    # Combined
    uncertainty_cfg = config.get("annotation", {}).get("uncertainty", {})
    weights = None
    signal_weights = uncertainty_cfg.get("weights")
    if signal_weights:
        weights = signal_weights

    combined = compute_combined_uncertainty(
        semantic, spatial, agreement, taxonomic, weights
    )
    print(f"    Combined:                 mean={combined.mean():.3f}")

    print("[annotate] Step 4/4: Classifying confidence tiers ...")
    tiers = classify_confidence(combined, config)

    tier_counts = Counter(tiers)
    for tier in ["auto_accept", "review", "human_required"]:
        pct = tier_counts.get(tier, 0) / n * 100
        print(f"    {tier:<20s} {tier_counts.get(tier, 0):5d}  ({pct:.1f}%)")

    # Build output DataFrame
    # Both ai_annotation and ai_second_choice come from CLUSTER-LEVEL centroid
    # matching (not per-image voting) so they are always consistent.
    # Per-image voting (image_top1) is used only for the agreement signal.
    df = pd.DataFrame({
        "file_path":              image_paths if image_paths else list(range(n)),
        "cluster_id":             cluster_ids,
        "ai_annotation":          [cluster_annotations[c]["species"] for c in cluster_ids],
        "ai_second_choice":       [cluster_annotations[c]["second_species"] for c in cluster_ids],
        "confidence":             [cluster_annotations[c]["confidence"] for c in cluster_ids],
        "margin":                 [cluster_annotations[c]["margin"] for c in cluster_ids],
        "semantic_uncertainty":   semantic,
        "spatial_uncertainty":    spatial,
        "cluster_agreement":      agreement,
        "taxonomic_consistency":  taxonomic,
        "combined_uncertainty":   combined,
        "confidence_tier":        tiers,
        "annotation_source":      "AI_ZeroShot",
    })

    return df
