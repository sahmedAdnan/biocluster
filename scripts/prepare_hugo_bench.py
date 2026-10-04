"""
prepare_hugo_bench.py
---------------------
Download the AI-EcoNet/HUGO-Bench dataset from HuggingFace and materialise it
as an image folder that Bio Cluster can open directly.

HUGO-Bench schema (config "default"):
    splits       : aves (73,528 birds) | mammals (65,583)
    image_bytes  : raw JPEG bytes
    species_name : ground-truth label (e.g. "Panthera leo")
    bbox         : camera-trap crop box "[x, y, w, h]" (optional use)

Output layout (labelled mode: one subfolder per species):
    <out>/<split>/<species_name>/000000.jpg
    <out>/<split>/<species_name>/000001.jpg
    ...

Then in Bio Cluster:
    Browse -> select the split folder (e.g. ...\\hugo_bench\\aves)
    tick "has labels", pick "BioCLIP 2.5 Huge", Run.

Usage:
    pip install datasets pillow
    python scripts/prepare_hugo_bench.py --split aves --out "/path/to/hugo_bench"
    python scripts/prepare_hugo_bench.py --split mammals --limit 2000   # quick test
    python scripts/prepare_hugo_bench.py --split aves --crop            # crop to bbox
"""

import argparse
import csv
import io
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

REPO = "AI-EcoNet/HUGO-Bench"


def safe_name(s: str) -> str:
    """Turn a species name into a filesystem-safe folder name."""
    s = (s or "unknown").strip()
    s = re.sub(r"[^A-Za-z0-9 _.-]", "_", s)   # drop path separators / oddities
    s = re.sub(r"\s+", "_", s)                # spaces -> underscores
    return s or "unknown"


def parse_bbox(raw):
    """bbox is stored as a string like '[x, y, w, h]'. Return (x,y,w,h) or None."""
    if not raw:
        return None
    try:
        vals = json.loads(raw) if isinstance(raw, str) else raw
        x, y, w, h = (float(v) for v in vals[:4])
        return x, y, w, h
    except Exception:
        return None


def fetch_url(session, url, timeout, retries):
    """GET an image with retries + backoff. Returns bytes, or raises."""
    last = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout,
                            headers={"User-Agent": "BioCluster/1.0 (research)"})
            r.raise_for_status()
            return r.content
        except Exception as e:            # link rot is common in these datasets
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


def main():
    ap = argparse.ArgumentParser(description="Materialise HUGO-Bench as an image folder.")
    ap.add_argument("--split", choices=["aves", "mammals", "both"], default="aves")
    ap.add_argument("--out", default="./data/hugo_bench",
                    help="Output root directory.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images per split (0 = all). Use a small value to test.")
    ap.add_argument("--crop", action="store_true",
                    help="Crop each image to its bbox before saving (tighter subjects).")
    ap.add_argument("--min-per-class", type=int, default=1,
                    help="Skip species with fewer than N images (0 = keep all).")
    ap.add_argument("--max-classes", type=int, default=0,
                    help="Keep only the first N species encountered (0 = all). "
                         "The dataset is streamed, so 'first' is dataset order, "
                         "not alphabetical -- use --species to name them exactly.")
    ap.add_argument("--species", default=None,
                    help="Comma-separated species to keep, e.g. "
                         "\"Corvus brachyrhynchos,Porphyrio melanotus\". Matched "
                         "against the folder-safe name, case-insensitively.")
    ap.add_argument("--per-class-limit", type=int, default=0,
                    help="Stop collecting a species after N images (0 = all). "
                         "With --max-classes or --species, streaming stops as "
                         "soon as every selected species is full.")
    ap.add_argument("--source", choices=["bytes", "url"], default="bytes",
                    help="'bytes' uses the JPEG packed into the dataset (fast, "
                         "but whatever resolution the authors stored). 'url' "
                         "fetches original_url instead -- the true original "
                         "image, at full resolution.")
    ap.add_argument("--delay", type=float, default=0.1,
                    help="Seconds between URL fetches. Be polite: these resolve "
                         "to third-party hosts.")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="Per-request timeout in seconds (--source url).")
    ap.add_argument("--retries", type=int, default=2,
                    help="Retries per URL, with backoff (--source url).")
    args = ap.parse_args()

    wanted = None
    if args.species:
        wanted = {safe_name(x).lower() for x in args.species.split(",") if x.strip()}
        print(f"[filter] keeping only: {sorted(wanted)}")

    splits = ["aves", "mammals"] if args.split == "both" else [args.split]
    out_root = Path(args.out).expanduser().resolve()

    for split in splits:
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {split} -> {out_dir} ===")
        print("Streaming from HuggingFace (no full local cache copy)...")

        ds = load_dataset(REPO, split=split, streaming=True)

        session = None
        if args.source == "url":
            import requests
            session = requests.Session()
            print(f"[source] fetching original_url  (delay={args.delay}s, "
                  f"timeout={args.timeout}s, retries={args.retries})")

        # Provenance: what each saved file actually came from. Worth having at
        # download time -- original_url and dataset_name cannot be reconstructed
        # later, and dataset_name is what lets you check whether a cluster is
        # tracking species or acquisition source.
        prov_path = out_dir / "provenance.csv"
        prov = open(prov_path, "w", newline="", encoding="utf-8")
        pw = csv.writer(prov)
        pw.writerow(["file", "species", "dataset_name", "category",
                     "source_image", "bbox", "original_url", "source"])

        saved, failed, per_class = 0, 0, {}
        for i, row in enumerate(ds):
            if args.limit and saved >= args.limit:
                break

            # Decide whether to keep this row BEFORE decoding the JPEG --
            # decoding is the expensive part and most rows are discarded when
            # only a couple of species are wanted.
            cls = safe_name(row.get("species_name"))
            if wanted is not None and cls.lower() not in wanted:
                continue
            if (args.max_classes and cls not in per_class
                    and len(per_class) >= args.max_classes):
                continue
            if args.per_class_limit and per_class.get(cls, 0) >= args.per_class_limit:
                # This species is full. If every selected species is full too,
                # there is nothing left to find -- stop streaming.
                target = len(wanted) if wanted else args.max_classes
                if (target and len(per_class) >= target
                        and all(n >= args.per_class_limit for n in per_class.values())):
                    print("  all selected species full -- stopping early")
                    break
                continue

            try:
                if args.source == "url":
                    url = row.get("original_url")
                    if not url:
                        raise ValueError("row has no original_url")
                    raw = fetch_url(session, url, args.timeout, args.retries)
                    if args.delay:
                        time.sleep(args.delay)
                else:
                    raw = row["image_bytes"]
                img = Image.open(io.BytesIO(raw)).convert("RGB")

                if args.crop:
                    box = parse_bbox(row.get("bbox"))
                    if box:
                        x, y, w, h = box
                        img = img.crop((int(x), int(y), int(x + w), int(y + h)))

                cls_dir = out_dir / cls
                cls_dir.mkdir(exist_ok=True)
                fname = f"{saved:06d}.jpg"
                img.save(cls_dir / fname, "JPEG", quality=95)
                pw.writerow([f"{cls}/{fname}", row.get("species_name", ""),
                             row.get("dataset_name", ""), row.get("category", ""),
                             row.get("source_image", ""), row.get("bbox", ""),
                             row.get("original_url", ""), args.source])

                per_class[cls] = per_class.get(cls, 0) + 1
                saved += 1
                if saved % 1000 == 0:
                    print(f"  {saved} images -> {len(per_class)} species")
            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"  [warn] row {i} failed: {e}")

        # Optionally prune tiny classes (clustering is noisy on singletons).
        if args.min_per_class > 1:
            removed = 0
            for cls, n in list(per_class.items()):
                if n < args.min_per_class:
                    for p in (out_dir / cls).glob("*.jpg"):
                        p.unlink()
                    (out_dir / cls).rmdir()
                    removed += n
                    per_class.pop(cls)
            print(f"  pruned {removed} images from classes < {args.min_per_class}")

        prov.close()
        print(f"Done: {saved} images, {len(per_class)} species, {failed} failed.")
        print(f"Provenance written to {prov_path}")
        print(f"Point Bio Cluster at: {out_dir}")


if __name__ == "__main__":
    main()
