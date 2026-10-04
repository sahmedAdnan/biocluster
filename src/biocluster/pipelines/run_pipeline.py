"""
Unified runner: execute the full BioCLIP clustering pipeline in one command.

3 steps, in order:
    1. extract   -- extract image embeddings (BioCLIP-2, BioCLIP, CLIP, DINOv2)
    2. cluster   -- run ALL clustering methods (KMeans, Agglomerative, Spectral,
                    GMM, DBSCAN, HDBSCAN). Methods are defined in the YAML config.
    3. evaluate  -- evaluate all cluster results, generate publication figures

Each step is skippable. Features and 2D projections are cached on disk.

Usage:
    # Run everything from scratch
    biocluster-pipeline --config configs/paper/eumoth/backbone_bioclip2.yaml

    # Resume from clustering (features already extracted)
    biocluster-pipeline --config configs/paper/eumoth/backbone_bioclip2.yaml --skip-extract

    # Skip both density methods
    biocluster-pipeline --config configs/paper/eumoth/backbone_bioclip2.yaml --skip-density

    # Re-run only evaluation figures
    biocluster-pipeline --config configs/paper/eumoth/backbone_bioclip2.yaml \\
        --skip-extract --skip-cluster

Each CLI step also works standalone:
    biocluster-extract    --config configs/paper/eumoth/backbone_bioclip2.yaml
    biocluster-cluster    --config configs/paper/eumoth/backbone_bioclip2.yaml
    biocluster-evaluate  --config configs/paper/eumoth/backbone_bioclip2.yaml
"""

import argparse
import time

import biocluster.cli.cluster as step_cluster
import biocluster.cli.evaluate as step_evaluate

import biocluster.cli.extract as step_extract

def _banner(title: str) -> None:
    bar = "=" * 70
    print("\n" + bar)
    print("  " + title)
    print(bar)


def _run_step(label: str, fn, config_path: str, **kwargs) -> float:
    """Run one pipeline step. Print a header, time it, return elapsed seconds."""
    _banner("STEP: " + label)
    t0 = time.time()
    fn(config_path=config_path, **kwargs)
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print("\n  [{label}] finished in {m}m {s:.0f}s".format(
        label=label, m=mins, s=secs))
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full BioCLIP-2 clustering pipeline end to end.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/paper/eumoth/backbone_bioclip2.yaml)."
    )

    # Step-level skip flags
    parser.add_argument("--skip-extract",  action="store_true",
                        help="Skip Step 1: feature extraction.\n"
                             "Use if features.npz already exists.")
    parser.add_argument("--skip-cluster",  action="store_true",
                        help="Skip Step 2: all clustering methods.")
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="Skip Step 3: evaluation and figures.")

    # Density method flags (only relevant when step 2 runs)
    parser.add_argument("--skip-density",  action="store_true",
                        help="Within Step 2: skip all density methods\n"
                             "(DBSCAN and HDBSCAN).")
    parser.add_argument("--skip-dbscan",   action="store_true",
                        help="Within Step 2: skip DBSCAN only.")
    parser.add_argument("--skip-hdbscan",  action="store_true",
                        help="Within Step 2: skip HDBSCAN only.\n"
                             "Use if hdbscan is not installed.")

    args = parser.parse_args()

    wall_start = time.time()
    timings    = {}

    _banner("PIPELINE START")
    steps_active = []
    if not args.skip_extract:  steps_active.append("extract")
    if not args.skip_cluster:  steps_active.append("cluster")
    if not args.skip_evaluate: steps_active.append("evaluate")

    print("  Config : " + args.config)
    print("  Steps  : " + "  >  ".join(steps_active) if steps_active
          else "  Steps  : (all skipped)")

    # ---- Step 1: Extract ----
    if not args.skip_extract:
        timings["extract"] = _run_step(
            "1/3  Feature Extraction",
            step_extract.main,
            args.config,
        )
    else:
        print("\n  [skip] Step 1: Feature Extraction")

    # ---- Step 2: Cluster (ALL methods, including density) ----
    if not args.skip_cluster:
        cluster_kwargs = {
            "skip_density":  args.skip_density,
            "skip_dbscan":   args.skip_dbscan  or args.skip_density,
            "skip_hdbscan":  args.skip_hdbscan or args.skip_density,
        }
        timings["cluster"] = _run_step(
            "2/3  Clustering  (KMeans / Agglomerative / Spectral / DBSCAN / HDBSCAN)",
            step_cluster.main,
            args.config,
            **cluster_kwargs,
        )
    else:
        print("\n  [skip] Step 2: Clustering")

    # ---- Step 3: Evaluate ----
    if not args.skip_evaluate:
        timings["evaluate"] = _run_step(
            "3/3  Evaluation + Publication Figures",
            step_evaluate.main,
            args.config,
        )
    else:
        print("\n  [skip] Step 3: Evaluation")

    # ---- Final summary ----
    total = time.time() - wall_start
    _banner("PIPELINE COMPLETE")
    if timings:
        print("  Timing breakdown:")
        for step_name, secs in timings.items():
            m = int(secs // 60)
            s = secs % 60
            print("    {:<12s}: {}m {:.0f}s".format(step_name, m, s))
    total_m = int(total // 60)
    total_s = total % 60
    print("\n  Total wall time : {}m {:.0f}s".format(total_m, total_s))
    print("  All outputs in the output_dir defined in your config.")


if __name__ == "__main__":
    main()
