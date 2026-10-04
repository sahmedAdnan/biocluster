"""
scripts/pick_dbscan_eps.py
--------------------------
Pick DBSCAN's `eps` for one (dataset, model, reduction) triple, without
using the ground-truth labels to make the choice.

Why this script exists
----------------------
`eps` is an absolute radius in the space DBSCAN actually sees, i.e. AFTER the
reduction. That space is rebuilt from scratch for every embedding model and
every reduction setting, and UMAP does not preserve scale between fits. So the
values already in the paper (EU-Moth 0.13 on UMAP-20d, MothsDK 0.12 on UMAP-20d,
CUB-200 0.08 on PCA+UMAP-5d) carry no information about a new run: a BioCLIP-2.5
HUGO-bench embedding has its own density and needs its own eps.

The rule used here is the standard k-distance heuristic of Ester et al. (1996):
sort every point's distance to its k-th nearest neighbour, and take eps at the
knee of that curve. Points below the knee sit inside dense regions and become
core points; the sharp rise past the knee is the tail of outliers you want left
as noise.

k and min_samples
-----------------
sklearn's DBSCAN counts the point itself in min_samples, so a point is core when
it has min_samples - 1 OTHER points within eps. The matching k-distance is
therefore the (min_samples - 1)-th neighbour excluding self, which is what
this script computes. Getting this off by one shifts the knee.

Protocol
--------
Run this on ONE subsample (the dev run), freeze the eps it gives, and use that
same value for all ten evaluation runs, the same way MothsDK used eps=0.12
across all ten seeds. Re-tuning eps per seed on the labels would leak ground
truth into the "unsupervised" numbers and shrink the reported SD.

The ARI/NMI columns printed at the end are a sanity check to run AFTER you have
committed to the knee value, never the thing you select on.

Usage
-----
    python scripts/pick_dbscan_eps.py \
        --features ~/BioCluster/feature_cache/<fingerprint>.npz \
        --method pca+umap --pca-components 150 --n-components 5 \
        --n-neighbors 30 --min-dist 0.15 --metric cosine \
        --min-samples 5 --seed 42

    # list what is in the feature cache
    python scripts/pick_dbscan_eps.py --list-cache
"""

import argparse
import json
import os

import numpy as np

# biocluster.utils.io pulls in torch, which a plain analysis environment need
# not have, so features are read straight from the .npz here and the reduction
# is imported lazily inside main().

CACHE_DIR = os.path.join(os.path.expanduser("~"), "BioCluster", "feature_cache")


def load_npz(path: str):
    """Read (features, labels) from a features .npz. Same format as
    biocluster.utils.io.save_features writes, minus the torch dependency."""
    data = np.load(path, allow_pickle=True)
    labels = data["labels"] if "labels" in data.files else None
    print(f"[load] {path}  features={data['features'].shape}")
    return data["features"], (None if labels is None else np.asarray(labels))


# -------------------------------------------------------
# Feature cache lookup
# -------------------------------------------------------

def list_cache() -> None:
    """Print the feature cache index so a fingerprint can be copied out."""
    index_path = os.path.join(CACHE_DIR, "index.json")
    if not os.path.exists(index_path):
        print(f"No cache index at {index_path}")
        return
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    for key, meta in sorted(index.items(),
                            key=lambda kv: kv[1].get("created", "")):
        print(f"{key}  {str(meta.get('shape')):<14} "
              f"{meta.get('model', '?'):<12} {meta.get('folder', '')}")


# -------------------------------------------------------
# k-distance curve and its knee
# -------------------------------------------------------

def k_distances(reduced: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance from every point to its (min_samples - 1)-th nearest
    neighbour, self excluded: the exact quantity DBSCAN thresholds at eps
    when deciding whether a point is core."""
    from sklearn.neighbors import NearestNeighbors

    k = max(1, min_samples - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(reduced)
    dist, _ = nn.kneighbors(reduced)
    return np.sort(dist[:, k])          # column 0 is the point itself


def knee_point(curve: np.ndarray, trim_pct: float = 97.5) -> tuple:
    """Knee of a sorted, rising curve: the point furthest below the chord
    joining its two endpoints (the Kneedle construction of Satopaa et al. 2011,
    computed directly rather than via a dependency).

    `trim_pct` drops the top tail of the curve before the search. This is not
    cosmetic. The chord is normalised by the curve's own range, and on these
    embeddings the last percentile runs 3-5x higher than the 95th (a handful of
    far-flung images), so an untrimmed curve puts the endpoint so high that the
    knee collapses towards zero. On HUGO-bench it returned eps=0.027, which
    labels 98% of the data as noise. Trimming at p97.5 removes that leverage
    without touching the part of the curve the knee actually sits in.

    Returns (index into the trimmed curve, eps value)."""
    c = curve[curve <= np.percentile(curve, trim_pct)] if trim_pct < 100 else curve
    n = len(c)
    x = np.linspace(0.0, 1.0, n)
    y = (c - c[0]) / (c[-1] - c[0] + 1e-12)
    idx = int(np.argmax(x - y))          # rising curve: elbow is max(x - y)
    return idx, float(c[idx])


def sweep(reduced: np.ndarray, eps_values, min_samples: int,
          labels=None) -> list:
    """Run DBSCAN at each eps. Reports the label-free diagnostics first; the
    supervised columns are filled only when labels are available."""
    from sklearn.cluster import DBSCAN

    rows = []
    for eps in eps_values:
        ids = DBSCAN(eps=float(eps), min_samples=min_samples,
                     metric="euclidean", n_jobs=-1).fit_predict(reduced)
        noise = ids == -1
        n_clusters = len(set(ids[~noise]))
        sizes = np.bincount(ids[~noise]) if n_clusters else np.array([0])
        row = {
            "eps": float(eps),
            "clusters": n_clusters,
            "noise_pct": 100.0 * noise.mean(),
            "largest_pct": 100.0 * sizes.max() / max(1, len(ids)),
        }
        if labels is not None and (~noise).sum() > 1:
            from sklearn.metrics import (adjusted_rand_score,
                                         normalized_mutual_info_score)
            y = np.asarray(labels)[~noise]
            p = ids[~noise]
            row["ARI"] = adjusted_rand_score(y, p)
            row["NMI"] = normalized_mutual_info_score(y, p)
        rows.append(row)
    return rows


def print_sweep(rows: list, knee_eps: float) -> None:
    has_eval = "ARI" in rows[0]
    head = f"{'eps':>8} {'clusters':>9} {'noise %':>8} {'largest %':>10}"
    if has_eval:
        head += f" {'ARI':>7} {'NMI':>7}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        line = (f"{r['eps']:>8.4f} {r['clusters']:>9d} "
                f"{r['noise_pct']:>8.2f} {r['largest_pct']:>10.2f}")
        if has_eval:
            line += f" {r['ARI']:>7.3f} {r['NMI']:>7.3f}"
        if abs(r["eps"] - knee_eps) < 1e-9:
            line += "   <- knee"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Choose DBSCAN eps from the k-distance knee.",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--features", help="features .npz (feature_cache entry or "
                                       "biocluster-extract output)")
    ap.add_argument("--list-cache", action="store_true",
                    help="print the feature cache index and exit")

    ap.add_argument("--method", default="pca+umap",
                    choices=["none", "pca", "umap", "pca+umap"])
    ap.add_argument("--pca-components", type=int, default=150)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.15)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=42,
                    help="random_state for the reduction; use the same one "
                         "as the run this eps is for")

    ap.add_argument("--min-samples", type=int, default=5,
                    help="DBSCAN min_samples; the k-distance curve is computed "
                         "for k = min_samples - 1")
    ap.add_argument("--trim", type=float, default=97.5,
                    help="percentile at which the k-distance curve is trimmed "
                         "before the knee search (100 = no trim)")
    ap.add_argument("--save-reduced", default="",
                    help="cache the reduced embedding to this .npy")
    ap.add_argument("--load-reduced", default="",
                    help="reuse a cached reduced embedding, skipping PCA/UMAP")
    ap.add_argument("--span", type=float, default=0.4,
                    help="sweep from (1-span) to (1+span) times the knee")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--extra-eps", default="",
                    help="comma-separated extra eps values to include")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the ARI/NMI sanity columns even if labels exist")
    ap.add_argument("--save-curve", default="",
                    help="write the sorted k-distance curve to this .csv")
    args = ap.parse_args()

    if args.list_cache:
        list_cache()
        return
    if not args.features:
        ap.error("--features is required (or use --list-cache)")

    features, labels = load_npz(os.path.expanduser(args.features))

    config = {"reduction": {
        "method":         args.method,
        "n_components":   args.n_components,
        "pca_components": args.pca_components,
        "n_neighbors":    args.n_neighbors,
        "min_dist":       args.min_dist,
        "metric":         args.metric,
    }}
    if args.load_reduced:
        reduced = np.load(os.path.expanduser(args.load_reduced))
        print(f"  [reduction] reusing cached embedding {reduced.shape}")
    else:
        from biocluster.reduction.reduction import reduce_features
        reduced, _ = reduce_features(features, config, random_state=args.seed)
    reduced = np.asarray(reduced, dtype=np.float64)
    if args.save_reduced:
        np.save(os.path.expanduser(args.save_reduced), reduced)

    curve = k_distances(reduced, args.min_samples)
    idx, knee = knee_point(curve, args.trim)

    print(f"\nk-distance curve (k = min_samples - 1 = {max(1, args.min_samples - 1)})")
    print(f"  n points        : {len(curve)}")
    print(f"  min / median / max: {curve[0]:.4f} / "
          f"{np.median(curve):.4f} / {curve[-1]:.4f}")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3d}            : {np.percentile(curve, q):.4f}")
    print(f"  knee            : eps = {knee:.4f} "
          f"(at point {idx}/{len(curve)}, "
          f"{100.0 * idx / len(curve):.1f}% of points below)")

    if args.save_curve:
        np.savetxt(args.save_curve, curve, delimiter=",",
                   header="k_distance", comments="")
        print(f"  curve written to {args.save_curve}")

    eps_values = list(np.linspace(knee * (1 - args.span),
                                  knee * (1 + args.span), args.steps))
    eps_values.append(knee)
    eps_values += [float(v) for v in args.extra_eps.split(",") if v.strip()]
    eps_values = sorted(set(round(v, 6) for v in eps_values))

    y = None if (args.no_eval or labels is None or not len(labels)) else labels
    print_sweep(sweep(reduced, eps_values, args.min_samples, y), round(knee, 6))

    print("\nTake the knee value, freeze it, and reuse it unchanged across "
          "every subsample run.\nThe ARI/NMI columns are a post-hoc check, not "
          "a selection criterion.")


if __name__ == "__main__":
    main()