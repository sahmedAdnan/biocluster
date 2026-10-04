"""
scripts/aggregate_results.py
----------------------------
Collapse the run-history log into a results table.

Reads the append-only run_history.csv (one row per completed run) and groups
runs that share a configuration (dataset, model, reduction and its params,
clustering and its params, device) but differ only by seed, reporting mean,
standard deviation, min, max and n for each metric. One row per
(dataset x configuration), which is the form the manuscript tables take.

The two subsampled datasets are reported as ten-run means, so this is how
those cells are produced.

run_history.csv is only ever read; the summary goes to a separate file.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --dataset cub200
    python scripts/aggregate_results.py --min-runs 3     # configs with >=3 seeds
    python scripts/aggregate_results.py --per-dataset    # also summary_<dataset>.csv

The log lives under the user's home directory by default; --history and --out
override both paths.
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

# Fields that DEFINE a configuration (grouped on). Seed/timestamp/metrics excluded.
KEY_FIELDS = ["dataset", "model", "device", "labeled",
              "reduction", "reduction_params", "clustering", "clustering_params"]

# Numeric outcomes to summarise (only those present in the CSV are used).
METRIC_FIELDS = ["ARI", "NMI", "Hungarian", "Silhouette",
                 "Calinski_Harabasz", "Davies_Bouldin", "n_clusters", "noise_pct"]

DEFAULT_HISTORY = os.path.join(os.path.expanduser("~"), "BioCluster", "run_history.csv")


def _fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _norm_params(s):
    """Normalise a params JSON string so identical params always match
    regardless of key order (e.g. '{\"b\":2,\"a\":1}' == '{\"a\":1,\"b\":2}')."""
    try:
        d = json.loads(s or "{}")
        return json.dumps(d, sort_keys=True)
    except Exception:
        return s or "{}"


def _stats(values):
    """mean/std/min/max/n for a list of floats (std=0 when n<2)."""
    n = len(values)
    if n == 0:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values), "n": n}


def aggregate(rows):
    """Group rows by configuration; return a list of summary dicts."""
    groups = defaultdict(list)
    for r in rows:
        key = tuple(
            _norm_params(r.get(f, "")) if f.endswith("_params") else r.get(f, "")
            for f in KEY_FIELDS
        )
        groups[key].append(r)

    summaries = []
    for key, grp in groups.items():
        summary = dict(zip(KEY_FIELDS, key))
        summary["n_runs"] = len(grp)
        # seeds that went into this config (for transparency)
        seeds = sorted({str(r.get("seed", "")) for r in grp if str(r.get("seed", "")) != ""})
        summary["seeds"] = ",".join(seeds)
        for m in METRIC_FIELDS:
            vals = [v for v in (_fnum(r.get(m)) for r in grp) if v is not None]
            st = _stats(vals)
            if st is None:
                continue
            summary[f"{m}_mean"] = round(st["mean"], 4)
            summary[f"{m}_std"] = round(st["std"], 4)
            summary[f"{m}_min"] = round(st["min"], 4)
            summary[f"{m}_max"] = round(st["max"], 4)
            summary[f"{m}_n"] = st["n"]
        summaries.append(summary)

    # Rank: by ARI_mean if any labeled, else Silhouette_mean, else 0.
    def sort_key(s):
        return (s.get("ARI_mean", s.get("Silhouette_mean", float("-inf"))),)
    summaries.sort(key=sort_key, reverse=True)
    return summaries


def write_summary(summaries, path):
    if not summaries:
        print("No configurations to summarise.")
        return
    # Union of all keys, ordered: config fields, meta, then metric stats.
    cols = list(KEY_FIELDS) + ["n_runs", "seeds"]
    for m in METRIC_FIELDS:
        for suffix in ("mean", "std", "min", "max", "n"):
            c = f"{m}_{suffix}"
            if any(c in s for s in summaries):
                cols.append(c)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"Wrote {len(summaries)} configuration summaries -> {path}")


def print_table(summaries):
    """Compact console view: best configs, paper-ready mean +/- std."""
    if not summaries:
        return
    metric = "ARI" if any("ARI_mean" in s for s in summaries) else "Silhouette"
    # Size the name columns to the actual content so nothing overflows.
    dw = max(10, min(30, max((len(s.get("dataset", "")) for s in summaries), default=10))) + 2
    mw = max(8, min(18, max((len(s.get("model", "")) for s in summaries), default=8))) + 2
    print(f"\n{'dataset':<{dw}}{'model':<{mw}}{'reduction':<10}{'clustering':<12}"
          f"{'n':>4}  {metric+' (mean +/- std)':<22}{'min':>7}{'max':>7}")
    print("-" * (dw + mw + 10 + 12 + 4 + 24 + 14))
    for s in summaries:
        mean = s.get(f"{metric}_mean")
        if mean is None:
            continue
        std = s.get(f"{metric}_std", 0.0)
        line = (f"{s.get('dataset',''):<{dw}}{s.get('model',''):<{mw}}"
                f"{s.get('reduction',''):<10}{s.get('clustering',''):<12}"
                f"{s.get('n_runs',0):>4}  {f'{mean:.3f} +/- {std:.3f}':<22}"
                f"{s.get(f'{metric}_min',0):>7.3f}{s.get(f'{metric}_max',0):>7.3f}")
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description="Summarise run_history.csv into a results table.")
    ap.add_argument("--history", default=DEFAULT_HISTORY, help="Path to run_history.csv.")
    ap.add_argument("--out", default=None, help="Summary CSV output path.")
    ap.add_argument("--dataset", default=None, help="Only summarise this dataset.")
    ap.add_argument("--min-runs", type=int, default=1, help="Only configs with >= this many runs.")
    ap.add_argument("--per-dataset", action="store_true",
                    help="Also write a separate summary_<dataset>.csv for each dataset.")
    args = ap.parse_args()

    if not os.path.exists(args.history):
        raise SystemExit(f"History file not found: {args.history}\n"
                         "Run the app at least once so it creates run_history.csv.")

    with open(args.history, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.dataset:
        rows = [r for r in rows if r.get("dataset") == args.dataset]
    print(f"Read {len(rows)} runs from {args.history}")

    summaries = aggregate(rows)
    if args.min_runs > 1:
        summaries = [s for s in summaries if s.get("n_runs", 0) >= args.min_runs]

    out = args.out or os.path.join(os.path.dirname(args.history), "run_history_summary.csv")
    write_summary(summaries, out)
    print_table(summaries)

    if args.per_dataset:
        by_ds = defaultdict(list)
        for s in summaries:
            by_ds[s.get("dataset", "unknown")].append(s)
        for ds, subs in by_ds.items():
            p = os.path.join(os.path.dirname(out), f"summary_{ds}.csv")
            write_summary(subs, p)


if __name__ == "__main__":
    main()