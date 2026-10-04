"""
Taxonomy lookup and caching for species annotation.

Supports three sources:
  1. Pre-computed CSV file (taxonomy_file in config)
  2. GBIF Species Match API (auto-query + local cache)
  3. Dataset class names only (genus parsed from name, no family/order)

The taxonomy table is a dict mapping class_name -> {genus, family, order}.
It is used by encode_text() to fill {family} and {order} prompt placeholders,
and by the annotator to validate taxonomic consistency across clusters.
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional

import requests

# -------------------------------------------------------
# Taxonomy Table Type
# -------------------------------------------------------

# Each entry: {"genus": str, "family": str, "order": str,
#              "match_type": str, "gbif_confidence": int}
TaxonomyEntry = Dict[str, str]
TaxonomyTable = Dict[str, TaxonomyEntry]


def load_taxonomy_csv(csv_path: str) -> TaxonomyTable:
    """
    Load taxonomy from a CSV file with columns:
        class_name, genus, family, order [, match_type, gbif_confidence]

    Returns:
        Dict mapping class_name -> taxonomy entry.
    """
    table = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("class_name", "").strip()
            if not name:
                continue
            table[name] = {
                "genus":  row.get("genus", ""),
                "family": row.get("family", ""),
                "order":  row.get("order", ""),
                "match_type": row.get("match_type", "CSV"),
                "gbif_confidence": int(row.get("gbif_confidence", 0) or 0),
            }

    print(f"[taxonomy] Loaded {len(table)} entries from {csv_path}")
    return table


# -------------------------------------------------------
# Source 2: GBIF Species Match API
# -------------------------------------------------------

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_DELAY     = 0.5       # seconds between requests (be gentle to GBIF API)
GBIF_RETRIES   = 3
GBIF_RETRY_DELAY = 10.0


def _query_gbif_single(species_name: str, kingdom: str = "Animalia") -> TaxonomyEntry:
    """
    Query GBIF species/match for one species name.

    Returns a taxonomy entry dict. Falls back to empty strings on failure.
    """
    params = {
        "name":    species_name,
        "kingdom": kingdom,
        "rank":    "SPECIES",
    }

    for attempt in range(1, GBIF_RETRIES + 1):
        try:
            r = requests.get(GBIF_MATCH_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            match_type = data.get("matchType", "NONE")
            confidence = data.get("confidence", 0)

            if match_type == "NONE" or confidence < 70:
                print(f"    [warn] Low match for '{species_name}': "
                      f"matchType={match_type}, confidence={confidence}")

            return {
                "genus":            data.get("genus", ""),
                "family":           data.get("family", ""),
                "order":            data.get("order", ""),
                "match_type":       match_type,
                "gbif_confidence":  confidence,
            }

        except requests.exceptions.RequestException as e:
            if attempt < GBIF_RETRIES:
                print(f"    [retry {attempt}/{GBIF_RETRIES}] '{species_name}': {e}")
                time.sleep(GBIF_RETRY_DELAY)
            else:
                print(f"    [fail] '{species_name}' after {GBIF_RETRIES} attempts: {e}")
                return {
                    "genus": "", "family": "", "order": "",
                    "match_type": "ERROR", "gbif_confidence": 0,
                }


def query_gbif_taxonomy(class_names: List[str],
                        kingdom: str = "Animalia",
                        cache_dir: Optional[str] = None) -> TaxonomyTable:
    """
    Query GBIF for each class name and return a taxonomy table.

    If cache_dir is provided, results are cached to a JSON file so
    subsequent runs skip the API entirely.

    Args:
        class_names: List of species names to look up.
        kingdom:     GBIF kingdom filter (default: Animalia).
        cache_dir:   Directory to cache results. None to disable caching.

    Returns:
        TaxonomyTable mapping class_name -> taxonomy entry.
    """
    # Check cache first
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "taxonomy_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Check if all names are in cache
            missing = [n for n in class_names if n not in cached]
            if not missing:
                print(f"[taxonomy] All {len(class_names)} entries loaded from cache")
                return cached
            print(f"[taxonomy] Cache has {len(cached)} entries, "
                  f"{len(missing)} missing - querying GBIF for missing")
        else:
            cached = {}
            missing = class_names
    else:
        cached = {}
        missing = class_names

    n = len(missing)
    print(f"[taxonomy] Querying GBIF for {n} species "
          f"(~{n * GBIF_DELAY / 60:.1f} min) ...")

    table = dict(cached)
    unknown_count = 0

    for i, name in enumerate(missing):
        pct = (i + 1) / n * 100
        print(f"    [{i+1:3d}/{n}] {pct:5.1f}%  {name}", end=" ... ", flush=True)

        entry = _query_gbif_single(name, kingdom)
        table[name] = entry

        status = entry["family"] if entry["family"] else "??"
        print(f"{status}  (conf={entry['gbif_confidence']})")

        if not entry["family"]:
            unknown_count += 1

        time.sleep(GBIF_DELAY)

    # Save cache
    if cache_dir:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
        print(f"[taxonomy] Cache saved to {cache_path}")

    resolved = n - unknown_count
    print(f"[taxonomy] Resolved: {resolved}/{n} ({resolved/n*100:.1f}%)")
    if unknown_count > 0:
        print(f"[taxonomy] Unresolved: {unknown_count} species")

    return table


def parse_taxonomy_from_names(class_names: List[str]) -> TaxonomyTable:
    """
    Minimal taxonomy: parse genus from the first word of each name.
    Family and order remain empty (templates using them will be skipped).
    """
    table = {}
    for name in class_names:
        readable = name.replace("_", " ").strip()
        genus = readable.split()[0] if " " in readable else readable
        table[name] = {
            "genus": genus, "family": "", "order": "",
            "match_type": "PARSED", "gbif_confidence": 0,
        }
    print(f"[taxonomy] Parsed genus from {len(table)} class names (no family/order)")
    return table


def save_taxonomy_csv(table: TaxonomyTable, csv_path: str) -> None:
    """Save the taxonomy table to a CSV for reuse."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    fieldnames = ["class_name", "genus", "family", "order",
                  "match_type", "gbif_confidence"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, entry in sorted(table.items()):
            row = {"class_name": name}
            row.update(entry)
            writer.writerow(row)

    print(f"[taxonomy] Saved {len(table)} entries to {csv_path}")


def build_taxonomy_table(config: dict,
                         class_names: List[str]) -> TaxonomyTable:
    """
    Build a taxonomy table based on the annotation config.

    Config keys (under annotation):
        taxonomy_file:   Path to a pre-computed CSV (source 1).
        taxonomy_source: "gbif" to query GBIF API (source 2).
                         "names" to parse from class names only (source 3).
        gbif:
            kingdom:     GBIF kingdom filter (default: Animalia).
            cache_dir:   Directory to cache GBIF results.

    Priority: taxonomy_file > taxonomy_source > fallback to names.
    """
    ann_cfg = config.get("annotation", {})

    # Source 1: pre-computed CSV
    taxonomy_file = ann_cfg.get("taxonomy_file")
    if taxonomy_file and os.path.exists(taxonomy_file):
        return load_taxonomy_csv(taxonomy_file)

    # Source 2: GBIF API
    source = ann_cfg.get("taxonomy_source", "names")
    if source == "gbif":
        gbif_cfg  = ann_cfg.get("gbif", {})
        kingdom   = gbif_cfg.get("kingdom", "Animalia")
        cache_dir = gbif_cfg.get("cache_dir", ".cache/gbif")
        return query_gbif_taxonomy(class_names, kingdom=kingdom,
                                   cache_dir=cache_dir)

    # Source 3: parse from names (fallback)
    return parse_taxonomy_from_names(class_names)
