"""
Human review workflow: export uncertain images for expert review,
then merge expert corrections back into the master dataset.

Workflow:
    1. export_review_set()       -> copies uncertain images + generates CSV template
    2. [Expert reviews in HTML report or fills CSV]
    3. merge_expert_review()     -> merges corrections into annotations_final.csv
    4. export_sorted_species()   -> organizes images into species folders
    5. generate_summary()        -> summary statistics + species list

Output files:
    review_template.csv         -- template for expert to fill
    review_images/              -- copied images with informative filenames
    annotations_final.csv       -- master dataset with human corrections merged
    sorted_species/             -- images organized by final species name
    summary.txt                 -- species list + stats for publication
"""

import os
import shutil
from datetime import datetime

import pandas as pd

def export_review_set(annotations_df: pd.DataFrame,
                      output_dir: str,
                      config: dict,
                      include_sanity_check: bool = True,
                      n_sanity: int = 25,
                      ) -> str:
    """
    Export uncertain images and a review template CSV.

    Selects images with confidence_tier in ["review", "human_required"],
    optionally adds sanity-check images (high-confidence samples to verify
    the AI is working correctly).

    Args:
        annotations_df:      Output from run_annotation().
        output_dir:          Directory to write review files.
        config:              Full YAML config dict.
        include_sanity_check: Whether to add high-confidence sanity images.
        n_sanity:            Number of sanity check images to include.

    Returns:
        Path to the review template CSV.
    """
    review_dir = os.path.join(output_dir, "review_images")
    os.makedirs(review_dir, exist_ok=True)

    # Select images needing review
    needs_review = annotations_df[
        annotations_df["confidence_tier"].isin(["review", "human_required"])
    ].copy()
    needs_review["review_reason"] = needs_review["confidence_tier"].apply(
        lambda t: "High AI Confusion" if t == "human_required" else "Moderate Uncertainty"
    )

    # Optionally add sanity check images
    if include_sanity_check:
        confident = annotations_df[
            annotations_df["confidence_tier"] == "auto_accept"
        ]
        if len(confident) > 0:
            sanity = confident.sample(
                n=min(n_sanity, len(confident)),
                random_state=42
            ).copy()
            sanity["review_reason"] = "Sanity Check (High Confidence)"
            needs_review = pd.concat([needs_review, sanity], ignore_index=True)

    print(f"[review] Exporting {len(needs_review)} images for review:")
    print(f"    Human required : {(needs_review['review_reason'] == 'High AI Confusion').sum()}")
    print(f"    Review         : {(needs_review['review_reason'] == 'Moderate Uncertainty').sum()}")
    if include_sanity_check:
        print(f"    Sanity check   : {(needs_review['review_reason'] == 'Sanity Check (High Confidence)').sum()}")

    # Copy images with informative filenames and build review CSV
    review_rows = []
    copy_count = 0

    for _, row in needs_review.iterrows():
        original_path = row["file_path"]
        ai_guess_1 = str(row["ai_annotation"]).replace(" ", "_").replace("/", "-")
        ai_guess_2 = str(row["ai_second_choice"]).replace(" ", "_").replace("/", "-")
        uncertainty = row["combined_uncertainty"]
        original_filename = os.path.basename(original_path)

        # Prefix based on review reason
        if "Sanity" in row["review_reason"]:
            prefix = "SANITY"
        elif row["confidence_tier"] == "human_required":
            prefix = "CONFUSED"
        else:
            prefix = "REVIEW"

        smart_filename = (f"{prefix}_{uncertainty:.2f}___"
                          f"{ai_guess_1}_vs_{ai_guess_2}___"
                          f"{original_filename}")

        # Windows MAX_PATH guard: use absolute path for length check
        new_path = os.path.join(review_dir, smart_filename)
        abs_path = os.path.abspath(new_path)
        if len(abs_path) > 250:
            abs_review_dir = os.path.abspath(review_dir)
            ext = os.path.splitext(original_filename)[1]
            # Budget: 250 - dir - separators - prefix_score(~15) - ext
            budget = 250 - len(abs_review_dir) - 1 - 15 - len(ext)
            if budget < 30:
                budget = 30
            # Truncate species part, keep original filename short too
            orig_short = original_filename
            if len(original_filename) > budget // 2:
                orig_short = original_filename[:budget // 2 - 3] + "..." + ext
            species_part = f"{ai_guess_1}_vs_{ai_guess_2}"
            remaining = budget - len(orig_short) - 6  # 6 for ___ separators
            if remaining < 10:
                remaining = 10
            species_part = species_part[:remaining]
            smart_filename = (f"{prefix}_{uncertainty:.2f}___"
                              f"{species_part}___"
                              f"{orig_short}")
            new_path = os.path.join(review_dir, smart_filename)

        try:
            shutil.copy2(original_path, new_path)
            copy_count += 1
        except Exception as e:
            print(f"    [warn] Could not copy {original_path}: {e}")
            continue

        review_rows.append({
            "Review_Filename":        smart_filename,
            "Review_Reason":          row["review_reason"],
            "AI_Guess_1":             row["ai_annotation"],
            "AI_Guess_2":             row["ai_second_choice"],
            "Uncertainty_Score":      round(uncertainty, 4),
            "Expert_Corrected_Label": "",   # Expert fills this
            "Expert_Notes":           "",   # Expert fills this
            "Original_Filename":      original_filename,
        })

    # Save review template
    review_df = pd.DataFrame(review_rows)
    csv_path = os.path.join(output_dir, "review_template.csv")
    review_df.to_csv(csv_path, index=False)

    print(f"[review] Copied {copy_count} images to {review_dir}")
    print(f"[review] Review template saved to {csv_path}")

    return csv_path


# -------------------------------------------------------
# Merge Expert Corrections
# -------------------------------------------------------

def _detect_csv_format(expert_df: pd.DataFrame) -> str:
    """Auto-detect CSV format from column names."""
    cols = set(expert_df.columns)
    if "Image_Verdict" in cols and "Filename" in cols:
        return "per_image"
    if "Expert_Verdict" in cols and "Cluster_ID" in cols:
        return "cluster"
    if "Original_Filename" in cols:
        return "review_template"
    raise ValueError(
        f"Unrecognized CSV format. Columns: {list(expert_df.columns)}\n"
        f"Expected one of:\n"
        f"  - Per-image CSV:     Filename, Cluster_ID, AI_Prediction, Image_Verdict, Moved_To_Species\n"
        f"  - Cluster CSV:       Cluster_ID, AI_Prediction, Expert_Verdict, Expert_Corrected_Label, ...\n"
        f"  - Review template:   Original_Filename, Expert_Corrected_Label, Expert_Notes"
    )


def _apply_human_verified(master_df, mask, new_label):
    """Apply expert correction to matched rows."""
    master_df.loc[mask, "ai_annotation"] = new_label
    master_df.loc[mask, "annotation_source"] = "Human_Expert"
    master_df.loc[mask, "combined_uncertainty"] = 0.0
    master_df.loc[mask, "semantic_uncertainty"] = 0.0
    master_df.loc[mask, "spatial_uncertainty"] = 0.0
    master_df.loc[mask, "cluster_agreement"] = 0.0
    master_df.loc[mask, "taxonomic_consistency"] = 0.0
    master_df.loc[mask, "confidence_tier"] = "human_verified"


def _merge_per_image(master_df, expert_df):
    """Merge per-image CSV from HTML report."""
    updates = 0
    confirmed = 0

    for _, row in expert_df.iterrows():
        fname = str(row.get("Filename", "")).strip()
        verdict = str(row.get("Image_Verdict", "")).strip()
        # Handle both column names (old: Corrected_Species, new: Moved_To_Species)
        moved_to = str(row.get("Moved_To_Species",
                       row.get("Corrected_Species", ""))).strip()

        if not fname:
            continue

        mask = master_df["file_path"].str.endswith(fname)
        if mask.sum() == 0:
            continue

        if verdict == "correct":
            # Expert confirmed AI prediction
            master_df.loc[mask, "annotation_source"] = "Human_Confirmed"
            master_df.loc[mask, "confidence_tier"] = "human_verified"
            master_df.loc[mask, "combined_uncertainty"] = 0.0
            confirmed += 1

        elif verdict == "wrong":
            # Expert says wrong but didn't specify correct name
            master_df.loc[mask, "annotation_source"] = "Flagged_Wrong"
            master_df.loc[mask, "confidence_tier"] = "human_required"

        elif verdict == "move" and moved_to and moved_to.lower() != "nan":
            # Expert moved image to a different species
            _apply_human_verified(master_df, mask, moved_to)
            updates += 1

    return updates, confirmed


def _merge_cluster(master_df, expert_df):
    """Merge cluster-level CSV from HTML report."""
    updates = 0
    confirmed = 0

    for _, row in expert_df.iterrows():
        cid_raw = row.get("Cluster_ID", "")
        try:
            cid = int(cid_raw)
        except (ValueError, TypeError):
            continue

        verdict = str(row.get("Expert_Verdict", "")).strip()
        corrected = str(row.get("Expert_Corrected_Label", "")).strip()
        notes = str(row.get("Expert_Notes", "")).strip().lower()

        mask = master_df["cluster_id"] == cid
        if mask.sum() == 0:
            continue

        if "junk" in notes or "delete" in notes:
            master_df.loc[mask, "annotation_source"] = "Flagged_for_Deletion"
            master_df.loc[mask, "ai_annotation"] = "INVALID"

        elif verdict == "correct":
            master_df.loc[mask, "annotation_source"] = "Human_Confirmed"
            master_df.loc[mask, "confidence_tier"] = "human_verified"
            master_df.loc[mask, "combined_uncertainty"] = 0.0
            confirmed += 1

        elif verdict in ("wrong", "not_listed"):
            if corrected and corrected.lower() != "nan":
                _apply_human_verified(master_df, mask, corrected)
                updates += 1
            else:
                master_df.loc[mask, "annotation_source"] = "Flagged_Wrong"
                master_df.loc[mask, "confidence_tier"] = "human_required"

    return updates, confirmed


def _merge_review_template(master_df, expert_df):
    """Merge review_template CSV (original format)."""
    updates = 0
    junk_flagged = 0

    for _, row in expert_df.iterrows():
        orig_name = str(row.get("Original_Filename", "")).strip()
        expert_label = str(row.get("Expert_Corrected_Label", "")).strip()
        notes = str(row.get("Expert_Notes", "")).strip().lower()

        if not orig_name:
            continue

        mask = master_df["file_path"].str.endswith(orig_name)
        if mask.sum() == 0:
            continue

        if "junk" in notes or "delete" in notes:
            master_df.loc[mask, "annotation_source"] = "Flagged_for_Deletion"
            master_df.loc[mask, "ai_annotation"] = "INVALID"
            junk_flagged += 1
        elif expert_label and expert_label.lower() != "nan":
            _apply_human_verified(master_df, mask, expert_label)
            updates += 1

    return updates, junk_flagged


def merge_expert_review(annotations_df: pd.DataFrame,
                        expert_csv_path: str,
                        output_dir: str,
                        ) -> pd.DataFrame:
    """
    Merge expert corrections back into the master annotations.

    Auto-detects three CSV formats:

    1. Per-image CSV (from HTML report "Download Per-Image CSV"):
       Filename, Cluster_ID, AI_Prediction, Image_Verdict, Moved_To_Species

    2. Cluster CSV (from HTML report "Download Cluster Review CSV"):
       Cluster_ID, AI_Prediction, Expert_Verdict, Expert_Corrected_Label,
       Expert_Notes, Images_Correct, Images_Wrong, Images_Moved

    3. Review template CSV (from export_review_set):
       Original_Filename, Expert_Corrected_Label, Expert_Notes

    Multiple CSVs can be merged sequentially (per-image first, then cluster).

    Args:
        annotations_df:  Master annotations DataFrame.
        expert_csv_path: Path to the expert-filled CSV.
        output_dir:      Directory to save the final output.

    Returns:
        Updated DataFrame with expert corrections merged.
    """
    expert_df = pd.read_csv(expert_csv_path)
    master_df = annotations_df.copy()

    fmt = _detect_csv_format(expert_df)
    print(f"[merge] Loading expert reviews from {expert_csv_path}")
    print(f"[merge] Detected format: {fmt} ({len(expert_df)} entries)")

    if fmt == "per_image":
        updates, confirmed = _merge_per_image(master_df, expert_df)
        print("\n[merge] Per-image merge complete:")
        print(f"    Moved to new species : {updates}")
        print(f"    Confirmed correct    : {confirmed}")

    elif fmt == "cluster":
        updates, confirmed = _merge_cluster(master_df, expert_df)
        print("\n[merge] Cluster-level merge complete:")
        print(f"    Corrected clusters   : {updates}")
        print(f"    Confirmed correct    : {confirmed}")

    elif fmt == "review_template":
        updates, junk_flagged = _merge_review_template(master_df, expert_df)
        print("\n[merge] Review template merge complete:")
        print(f"    Expert corrections   : {updates}")
        print(f"    Flagged for delete   : {junk_flagged}")

    # Save final output
    final_path = os.path.join(output_dir, "annotations_final.csv")
    master_df.to_csv(final_path, index=False)

    # Provenance summary
    source_counts = master_df["annotation_source"].value_counts()
    print("\n    Provenance:")
    for source, count in source_counts.items():
        pct = count / len(master_df) * 100
        print(f"      {source:<25s} {count:5d}  ({pct:.1f}%)")

    print(f"\n[merge] Final dataset saved to {final_path}")
    return master_df


def export_sorted_species(final_df: pd.DataFrame,
                          output_dir: str,
                          copy: bool = True,
                          ) -> str:
    """
    Organize all images into folders by their final species label.

    Creates:
        sorted_species/
          Aglia_tau/
            img1.jpg
            img2.jpg
          Abrostola_tripartita/
            ...
          _FLAGGED/         (flagged for deletion)
          _NOISE/           (cluster_id == -1)

    Args:
        final_df:    DataFrame with file_path and ai_annotation columns.
        output_dir:  Base output directory.
        copy:        If True, copy files. If False, create symlinks.

    Returns:
        Path to the sorted_species directory.
    """
    sorted_dir = os.path.join(output_dir, "sorted_species")

    # Clean previous export
    if os.path.exists(sorted_dir):
        shutil.rmtree(sorted_dir)

    species_counts = {}

    for _, row in final_df.iterrows():
        src = row["file_path"]
        label = str(row["ai_annotation"]).strip()
        source = str(row.get("annotation_source", "")).strip()
        cid = row.get("cluster_id", 0)

        # Determine folder name.
        #
        # Routing priority. The final human or AI decision is the single
        # source of truth, NOT the original cluster_id. A human who moved
        # a noise image to a species means it belongs to that species, so
        # that decision must override a stale cluster_id == -1.
        #   1. Explicit deletion / junk        -> _FLAGGED
        #   2. Explicit human "noise"          -> _NOISE
        #   3. Human-assigned real species     -> species folder
        #      (this is the noise-rescue case that was previously lost)
        #   4. Still-unplaced AI noise         -> _NOISE
        #   5. Otherwise                       -> species folder by label
        human_labelled = ("Human_Expert", "Human_Confirmed")
        if label == "INVALID" or source == "Flagged_for_Deletion":
            folder_name = "_FLAGGED"
        elif source == "Human_Noise":
            folder_name = "_NOISE"
        elif (source in human_labelled
              and label and label.lower() not in ("nan", "invalid")):
            folder_name = label.replace(" ", "_")
        elif cid == -1:
            folder_name = "_NOISE"
        else:
            folder_name = label.replace(" ", "_")

        dest_dir = os.path.join(sorted_dir, folder_name)
        os.makedirs(dest_dir, exist_ok=True)

        fname = os.path.basename(src)
        dest = os.path.join(dest_dir, fname)

        # Handle duplicates
        if os.path.exists(dest):
            base, ext = os.path.splitext(fname)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                counter += 1

        try:
            if copy:
                shutil.copy2(src, dest)
            else:
                os.symlink(os.path.abspath(src), dest)
        except Exception as e:
            print(f"    [warn] Could not {'copy' if copy else 'link'} {src}: {e}")
            continue

        species_counts[folder_name] = species_counts.get(folder_name, 0) + 1

    # Print summary
    real_species = {k: v for k, v in species_counts.items()
                    if not k.startswith("_")}
    print(f"\n[export] Organized {sum(species_counts.values())} images "
          f"into {len(real_species)} species folders")
    if "_FLAGGED" in species_counts:
        print(f"    Flagged: {species_counts['_FLAGGED']} images")
    if "_NOISE" in species_counts:
        print(f"    Noise:   {species_counts['_NOISE']} images")
    print(f"[export] Output: {sorted_dir}")

    return sorted_dir


# -------------------------------------------------------
# Post-Merge: Summary Statistics
# -------------------------------------------------------

def generate_summary(final_df: pd.DataFrame,
                     output_dir: str,
                     config: dict = None,
                     ) -> str:
    """
    Generate summary statistics for publication / thesis.

    Creates:
        summary.txt   -- human-readable summary
        species_list.csv -- clean species list with counts

    Args:
        final_df:    Final annotations DataFrame.
        output_dir:  Base output directory.
        config:      Pipeline config for metadata.

    Returns:
        Path to the summary.txt file.
    """
    # Filter out flagged/invalid
    valid = final_df[final_df["ai_annotation"] != "INVALID"].copy()
    clustered = valid[valid["cluster_id"] >= 0]
    noise = valid[valid["cluster_id"] == -1]

    # Species stats
    species_counts = clustered.groupby("ai_annotation").agg(
        n_images=("file_path", "count"),
        n_clusters=("cluster_id", "nunique"),
        avg_uncertainty=("combined_uncertainty", "mean"),
    ).sort_values("n_images", ascending=False).reset_index()
    species_counts.rename(columns={"ai_annotation": "species"}, inplace=True)

    # Add provenance per species
    prov_counts = clustered.groupby(
        ["ai_annotation", "annotation_source"]
    ).size().unstack(fill_value=0).reset_index()
    prov_counts.rename(columns={"ai_annotation": "species"}, inplace=True)

    species_full = species_counts.merge(prov_counts, on="species", how="left")

    # Save species list CSV
    species_csv_path = os.path.join(output_dir, "species_list.csv")
    species_full.to_csv(species_csv_path, index=False)

    # Provenance breakdown
    prov = final_df["annotation_source"].value_counts()
    ai_count = prov.get("AI_ZeroShot", 0)
    confirmed_count = prov.get("Human_Confirmed", 0)
    corrected_count = prov.get("Human_Expert", 0)
    flagged_wrong = prov.get("Flagged_Wrong", 0)
    flagged_del = prov.get("Flagged_for_Deletion", 0)

    # Tier breakdown
    tiers = final_df["confidence_tier"].value_counts()

    # Correction rate
    total_reviewed = confirmed_count + corrected_count + flagged_wrong
    if total_reviewed > 0:
        ai_accuracy_on_reviewed = confirmed_count / total_reviewed * 100
    else:
        ai_accuracy_on_reviewed = None

    # Build summary text
    lines = []
    lines.append("=" * 60)
    lines.append("  SPECIES IDENTIFICATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if config:
        model = config.get("model", {}).get("name", "unknown")
        method = config.get("clustering", {}).get("method", "unknown")
        lines.append(f"  Model:     {model}")
        lines.append(f"  Clustering: {method}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  DATASET OVERVIEW")
    lines.append("-" * 60)
    lines.append(f"  Total images:       {len(final_df):>6d}")
    lines.append(f"  Clustered images:   {len(clustered):>6d}")
    lines.append(f"  Noise (unassigned): {len(noise):>6d}")
    lines.append(f"  Species identified: {species_counts['species'].nunique():>6d}")
    lines.append(f"  Clusters formed:    {clustered['cluster_id'].nunique():>6d}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  ANNOTATION PROVENANCE")
    lines.append("-" * 60)
    lines.append(f"  AI auto-accepted:     {ai_count:>6d}  "
                 f"({ai_count/len(final_df)*100:.1f}%)")
    lines.append(f"  Human confirmed:      {confirmed_count:>6d}  "
                 f"({confirmed_count/len(final_df)*100:.1f}%)")
    lines.append(f"  Human corrected:      {corrected_count:>6d}  "
                 f"({corrected_count/len(final_df)*100:.1f}%)")
    if flagged_wrong:
        lines.append(f"  Flagged (wrong):      {flagged_wrong:>6d}  "
                     f"({flagged_wrong/len(final_df)*100:.1f}%)")
    if flagged_del:
        lines.append(f"  Flagged (delete):     {flagged_del:>6d}  "
                     f"({flagged_del/len(final_df)*100:.1f}%)")
    lines.append("")

    if ai_accuracy_on_reviewed is not None:
        lines.append("-" * 60)
        lines.append("  AI PERFORMANCE (on human-reviewed subset)")
        lines.append("-" * 60)
        lines.append(f"  Images reviewed:    {total_reviewed:>6d}")
        lines.append(f"  AI was correct:     {confirmed_count:>6d}  "
                     f"({ai_accuracy_on_reviewed:.1f}%)")
        lines.append(f"  AI was wrong:       {corrected_count + flagged_wrong:>6d}  "
                     f"({(corrected_count + flagged_wrong)/total_reviewed*100:.1f}%)")
        lines.append("")

    lines.append("-" * 60)
    lines.append("  CONFIDENCE TIERS")
    lines.append("-" * 60)
    for tier in ["human_verified", "auto_accept", "review", "human_required"]:
        count = tiers.get(tier, 0)
        if count > 0:
            lines.append(f"  {tier:<20s}  {count:>6d}  "
                         f"({count/len(final_df)*100:.1f}%)")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  SPECIES LIST (sorted by image count)")
    lines.append("-" * 60)
    for _, sp in species_counts.iterrows():
        prov_tag = ""
        sp_prov = final_df[final_df["ai_annotation"] == sp["species"]]
        if "Human_Expert" in sp_prov["annotation_source"].values:
            prov_tag = " [corrected]"
        elif "Human_Confirmed" in sp_prov["annotation_source"].values:
            prov_tag = " [confirmed]"
        lines.append(f"  {sp['species']:<40s}  {int(sp['n_images']):>4d} images  "
                     f"(unc: {sp['avg_uncertainty']:.2f}){prov_tag}")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  Species list CSV: {species_csv_path}")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)

    # Save
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Print to console
    print(f"\n{summary_text}")
    print(f"\n[summary] Saved to {summary_path}")
    print(f"[summary] Species list: {species_csv_path}")

    return summary_path
