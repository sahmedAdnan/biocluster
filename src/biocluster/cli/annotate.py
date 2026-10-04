"""
CLI command: Annotate clusters with species names via zero-shot text matching.


Two sub-commands:

    annotate  --  Run annotation pipeline (cluster -> text match -> uncertainty)
    merge     --  Merge expert corrections from review CSV

Usage:
    # Run annotation on a specific clustering method:
    biocluster-annotate --config configs/paper/dkmoth_sub/backbone_bioclip2.yaml --method hdbscan

    # Merge expert review corrections:
    biocluster-annotate --config configs/paper/dkmoth_sub/backbone_bioclip2.yaml --merge review_template_filled.csv

Requires:
    <output_dir>/features.npz              (from extract.py)
    <output_dir>/clusters_<method>.npz     (from cluster.py)

Output:
    <output_dir>/annotations_<method>.csv      -- full annotation results
    <output_dir>/review_template.csv           -- template for human expert
    <output_dir>/review_images/                -- copied uncertain images
    <output_dir>/annotations_final.csv         -- after expert merge
    <output_dir>/figures/uncertainty_*.png      -- visualization plots
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

from biocluster.annotation.annotator import run_annotation
from biocluster.annotation.gallery import generate_gallery
from biocluster.annotation.review import export_review_set, merge_expert_review
from biocluster.annotation.taxonomy import build_taxonomy_table
from biocluster.utils.io import (
    ensure_dir,
    get_2d_projection,
    load_clusters,
    load_config,
    load_features,
    set_seed,
)
from biocluster.utils.pretty import done, header, metrics_table, saved, subheader

def _load_hdbscan_probabilities(output_dir: str, method_name: str,
                                 n_samples: int):
    """
    Try to load HDBSCAN membership probabilities from the pipeline output.

    The HDBSCANPipeline saves probabilities in the metrics JSON.
    Returns None if not available.
    """
    if method_name != "hdbscan":
        return None

    probs_path = os.path.join(output_dir, "hdbscan_probabilities.npy")
    if os.path.exists(probs_path):
        probs = np.load(probs_path)
        if len(probs) == n_samples:
            print(f"    [load] HDBSCAN probabilities from {probs_path}")
            return probs

    return None


# -------------------------------------------------------
# Main: Annotate
# -------------------------------------------------------

def main(config_path: str = None) -> None:
    parser = argparse.ArgumentParser(
        description="Annotate clusters with species names (Step 3)."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--report", action="store_true",
                        help="Also write the self-contained interactive review "
                             "HTML. Can be large: it embeds every thumbnail, "
                             "about 1 MB per 60 images.")
    parser.add_argument("--method", default=None,
                        help="Clustering method to annotate (e.g., 'hdbscan'). "
                             "If not specified, annotates all available methods.")
    parser.add_argument("--merge", default=None, metavar="CSV_PATH",
                        help="Path to expert-filled review CSV to merge.")

    if config_path is not None:
        # Called programmatically
        args = argparse.Namespace(config=config_path, method=None, merge=None,
                                  report=False)
    else:
        args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "figures"))

    # ---- Merge mode ----
    if args.merge:
        header("ANNOTATION — Merge Expert Review")

        # Find the latest annotations CSV
        method = args.method
        if method:
            ann_path = os.path.join(output_dir, f"annotations_{method}.csv")
        else:
            # Find any annotations CSV
            import glob
            csvs = sorted(glob.glob(os.path.join(output_dir, "annotations_*.csv")))
            if not csvs:
                print("ERROR: No annotations CSV found. Run annotate first.")
                sys.exit(1)
            ann_path = csvs[-1]
            method = os.path.basename(ann_path).replace("annotations_", "").replace(".csv", "")

        import pandas as pd
        annotations_df = pd.read_csv(ann_path)
        print(f"    Loaded annotations from {ann_path}")

        method_output_dir = os.path.join(output_dir, f"annotation_{method}")
        ensure_dir(method_output_dir)

        final_df = merge_expert_review(annotations_df, args.merge, method_output_dir)

        # Regenerate gallery with provenance info
        features_path = os.path.join(output_dir, "features.npz")
        features, _, _, _ = load_features(features_path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="n_jobs value.*overridden", category=UserWarning)
            features_2d = get_2d_projection(features, config, output_dir)

        generate_gallery(final_df, features_2d, method_output_dir, config,
                         method_name=method)
        done(output_dir)
        return

    # ---- Annotate mode ----
    header("STEP: Annotation")

    # Load features
    features_path = os.path.join(output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"\n  ERROR: features.npz not found at {features_path}")
        print("  Run first:  biocluster-extract  --config <config>")
        sys.exit(1)

    features, labels, image_paths, class_names = load_features(features_path)
    print(f"    Dataset : {len(image_paths)} images, {len(class_names)} classes")
    print(f"    Dim     : {features.shape[1]}D")

    # Determine which methods to annotate
    if args.method:
        methods_to_annotate = [args.method]
    else:
        # Find all cluster files
        import glob
        cluster_files = glob.glob(os.path.join(output_dir, "clusters_*.npz"))
        methods_to_annotate = [
            os.path.basename(f).replace("clusters_", "").replace(".npz", "")
            for f in cluster_files
        ]

    if not methods_to_annotate:
        print("ERROR: No cluster files found. Run cluster.py first.")
        sys.exit(1)

    print(f"    Methods : {', '.join(methods_to_annotate)}")

    # ---- Build taxonomy table ----
    subheader("Taxonomy Lookup")
    taxonomy_table = build_taxonomy_table(config, class_names)

    # ---- Build text embeddings ----
    subheader("Text Embedding (Species Dictionary)")

    # Get species list for annotation
    ann_cfg = config.get("annotation", {})
    species_source = ann_cfg.get("species_source", "dataset")

    if species_source == "dataset":
        species_names = list(class_names)
        print(f"    Species source: dataset class names ({len(species_names)} species)")
    elif species_source == "file":
        species_file = ann_cfg.get("species_file")
        if species_file and os.path.exists(species_file):
            import pandas as pd
            sdf = pd.read_csv(species_file)
            col = "class_name" if "class_name" in sdf.columns else sdf.columns[0]
            species_names = sdf[col].tolist()
            print(f"    Species source: {species_file} ({len(species_names)} species)")
        else:
            print(f"    [warn] Species file not found: {species_file}, "
                  f"falling back to dataset class names")
            species_names = list(class_names)
    elif species_source == "gbif":
        # Query GBIF for species occurrences
        species_names = _fetch_gbif_species_list(ann_cfg)
        print(f"    Species source: GBIF query ({len(species_names)} species)")
    else:
        species_names = list(class_names)

    # Encode text embeddings with taxonomy-enriched prompts
    from biocluster.models.extractors import build_extractor
    extractor = build_extractor(config)
    templates = config["model"].get("prompt_templates", None)
    text_embs = extractor.encode_text(
        species_names,
        prompt_templates=templates,
        taxonomy_table=taxonomy_table,
    )

    # ---- Pre-compute 2D projection for visualization ----
    subheader("2D Projection")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message="n_jobs value.*overridden", category=UserWarning)
        features_2d = get_2d_projection(features, config, output_dir)

    # ---- Annotate each method ----
    for method_name in methods_to_annotate:
        cluster_path = os.path.join(output_dir, f"clusters_{method_name}.npz")
        if not os.path.exists(cluster_path):
            print(f"\n    [skip] {method_name}: cluster file not found")
            continue

        subheader(f"Annotating: {method_name.upper()}")
        t0 = time.time()

        cluster_ids, _, _ = load_clusters(cluster_path)

        # Try to load HDBSCAN probabilities
        probabilities = _load_hdbscan_probabilities(
            output_dir, method_name, len(features)
        )

        # Run annotation pipeline
        annotations_df = run_annotation(
            features=features,
            cluster_ids=cluster_ids,
            text_embs=text_embs,
            species_names=species_names,
            taxonomy_table=taxonomy_table,
            config=config,
            probabilities=probabilities,
            image_paths=image_paths,
        )

        # Add ground truth column for accuracy computation
        ground_truth = [class_names[l] for l in labels]
        annotations_df["ground_truth"] = ground_truth

        # Compute annotation accuracy (AI label vs ground truth)
        valid_mask = cluster_ids >= 0  # exclude noise
        if valid_mask.any():
            correct = annotations_df.loc[valid_mask, "ai_annotation"] == \
                      annotations_df.loc[valid_mask, "ground_truth"]
            annotation_acc = correct.sum() / valid_mask.sum()
        else:
            annotation_acc = 0.0

        # Save annotations
        ann_path = os.path.join(output_dir, f"annotations_{method_name}.csv")
        annotations_df.to_csv(ann_path, index=False)
        saved(f"annotations_{method_name}.csv", ann_path)

        # Interactive review HTML, the same artefact the desktop app's
        # "Generate Web Report" button produces. Opt-in because it embeds every
        # thumbnail and so grows with the dataset.
        if getattr(args, "report", False) or ann_cfg.get("interactive_report"):
            from biocluster.annotation.interactive_report import (
                generate_interactive_report)
            reference_map = _build_reference_map(
                ann_cfg, annotations_df, image_paths, labels, class_names)
            mode = {"file": "csv", "gbif": "gbif"}.get(species_source, "local")
            report_path = os.path.join(output_dir, f"review_{method_name}.html")
            generate_interactive_report(annotations_df, reference_map, report_path,
                                        config=config, candidate_mode=mode)
            saved(f"review_{method_name}.html", report_path)

        elapsed = time.time() - t0

        # Summary metrics
        n_clusters = len(set(cluster_ids[cluster_ids >= 0]))
        n_auto = (annotations_df["confidence_tier"] == "auto_accept").sum()
        n_review = (annotations_df["confidence_tier"] == "review").sum()
        n_human = (annotations_df["confidence_tier"] == "human_required").sum()
        n_noise = int((cluster_ids == -1).sum())

        metrics_table([
            {"label": "Clusters annotated", "value": n_clusters},
            {"label": "Annotation accuracy", "value": f"{annotation_acc:.1%}",
             "note": f"(AI label vs ground truth, {int(correct.sum())}/{int(valid_mask.sum())})"},
            {"label": "Auto accepted",      "value": n_auto,
             "note": f"({n_auto/len(annotations_df)*100:.1f}%)"},
            {"label": "Needs review",       "value": n_review,
             "note": f"({n_review/len(annotations_df)*100:.1f}%)"},
            {"label": "Human required",     "value": n_human,
             "note": f"({n_human/len(annotations_df)*100:.1f}%)"},
            {"label": "Noise points",       "value": n_noise,
             "note": f"({n_noise/len(annotations_df)*100:.1f}%)"},
            {"label": "Time",              "value": f"{elapsed:.1f}s"},
        ], title=f"Annotation — {method_name}")

        # Per-cluster accuracy breakdown
        if valid_mask.any():
            subheader(f"Per-cluster accuracy: {method_name.upper()}")
            cluster_stats = []
            for cid in sorted(set(cluster_ids[valid_mask])):
                c_mask = cluster_ids == cid
                c_gt = annotations_df.loc[c_mask, "ground_truth"]
                c_ai = annotations_df.loc[c_mask, "ai_annotation"]
                c_correct = (c_ai == c_gt).sum()
                c_total = c_mask.sum()
                c_acc = c_correct / c_total
                c_species = c_ai.iloc[0]
                c_unc = annotations_df.loc[c_mask, "combined_uncertainty"].mean()
                cluster_stats.append({
                    "cluster": cid, "species": c_species,
                    "acc": c_acc, "correct": c_correct,
                    "total": c_total, "uncertainty": c_unc,
                })

            # Sort by accuracy (worst first) for easy review
            cluster_stats.sort(key=lambda x: x["acc"])
            print(f"    {'Cluster':>8s}  {'Accuracy':>8s}  {'Size':>5s}  "
                  f"{'Uncert':>6s}  Species")
            print(f"    {'-------':>8s}  {'--------':>8s}  {'----':>5s}  "
                  f"{'------':>6s}  -------")
            for s in cluster_stats:
                print(f"    {s['cluster']:>8d}  {s['acc']:>8.1%}  "
                      f"{s['total']:>5d}  {s['uncertainty']:>6.3f}  "
                      f"{s['species']}")

        # Export review set (per-method subdirectory)
        subheader(f"Exporting review set: {method_name.upper()}")
        method_output_dir = os.path.join(output_dir, f"annotation_{method_name}")
        ensure_dir(method_output_dir)
        export_review_set(annotations_df, method_output_dir, config)

        # Generate gallery (per-method figures)
        subheader(f"Gallery: {method_name.upper()}")
        generate_gallery(annotations_df, features_2d, method_output_dir, config,
                         method_name=method_name)

    done(output_dir)


def _build_reference_map(ann_cfg: dict, annotations_df, image_paths,
                         labels, class_names) -> dict:
    """
    Build the species -> reference images map the review report displays
    beside each cluster.

    Two sources, selected by annotation.reference_source:

      dataset  Images from the collection's own class directories. Available
               only on a labelled collection. Useful for inspecting what the
               clustering did, but it is not an independent check of a
               proposed name, because the proposal came from these images.

      gbif     Photographs of the proposed species fetched from GBIF. This is
               the independent comparison, and the only option that works on
               an unlabelled collection. Needs network access, and the report
               needs it again when opened, since only URLs are stored.
    """
    source = ann_cfg.get("reference_source", "dataset")

    if source == "gbif":
        from biocluster.annotation.gbif_media import (ADULT_STAGES,
                                                      collect_reference_urls)
        from biocluster.annotation.interactive_report import displayed_species
        gbif_cfg = ann_cfg.get("gbif", {})
        return collect_reference_urls(
            displayed_species(annotations_df),
            images_per_species=gbif_cfg.get("reference_images", 3),
            cache_dir=gbif_cfg.get("cache_dir", ".cache/gbif"),
            kingdom=gbif_cfg.get("kingdom", "Animalia"),
            life_stages=tuple(gbif_cfg.get("life_stages", ADULT_STAGES)),
        )

    if labels is None or class_names is None:
        print("    [warn] reference_source 'dataset' needs ground-truth class "
              "directories; the report will show no reference panels. Set "
              "annotation.reference_source: gbif to compare against GBIF "
              "photographs instead.")
        return {}

    reference_map = {}
    for path, label in zip(image_paths, labels):
        reference_map.setdefault(str(class_names[int(label)]), []).append(path)
    return reference_map


def _fetch_gbif_species_list(ann_cfg: dict) -> list:
    """
    Fetch species names from GBIF occurrence facets.

    Config keys (under annotation.gbif):
        taxon_key: int    -- GBIF taxon key (e.g., 797 for Lepidoptera)
        continent: str    -- continent filter (e.g., "EUROPE")
        country:   str    -- country code (e.g., "DK")
        limit:     int    -- max species to fetch (default: 500)
    """
    import requests

    gbif_cfg = ann_cfg.get("gbif", {})
    taxon_key = gbif_cfg.get("taxon_key", 797)
    limit     = gbif_cfg.get("limit", 500)

    params = {
        "taxonKey":   taxon_key,
        "limit":      0,
        "facet":      "scientificName",
        "facetLimit": limit,
    }

    continent = gbif_cfg.get("continent")
    country   = gbif_cfg.get("country")
    if continent:
        params["continent"] = continent
    if country:
        params["country"] = country

    url = "https://api.gbif.org/v1/occurrence/search"
    print(f"    Querying GBIF: taxonKey={taxon_key}, limit={limit}")

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        species_set = set()
        if "facets" in data and data["facets"]:
            for item in data["facets"][0].get("counts", []):
                parts = item.get("name", "").split()
                if (len(parts) >= 2
                        and "sp." not in parts[1]
                        and "indet" not in parts[1].lower()):
                    species_set.add(f"{parts[0]} {parts[1]}")

        return sorted(species_set)

    except Exception as e:
        print(f"    [error] GBIF query failed: {e}")
        return []


if __name__ == "__main__":
    main()
