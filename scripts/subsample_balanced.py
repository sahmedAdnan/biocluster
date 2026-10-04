"""
subsample_balanced.py
----------------------
Draw a class-balanced random subset from an already-materialised image folder
(one subfolder per class), reproducing the HUGO-Bench paper's evaluation protocol:
N randomly sampled images per species, with a fixed seed per run.

The paper (arXiv:2602.03894) never clustered the full corpus at once. Each run
used 200 images/species (30 species -> 6,000 images), repeated over 10 independent
random subsets to measure variance. Change the seed to get a different run.

Layout in and out (labelled mode: subfolder = class):
    <src>/<species>/*.jpg   ->   <out>/<species>/*.jpg   (<= N per species)

Usage:
    # Paper protocol: 200 birds/species, seed 0
    python scripts/subsample_balanced.py --src "/path/to/hugo_bench/aves" ^
        --per-class 200 --seed 0 --out "/path/to/hugo_bench/aves_200_run0"

    # A second independent run (different random subset)
    python scripts/subsample_balanced.py --src "/path/to/hugo_bench/aves" ^
        --per-class 200 --seed 1 --out "/path/to/hugo_bench/aves_200_run1"

    # Symlink instead of copy to save disk (needs admin/developer mode on Windows)
    python scripts/subsample_balanced.py --src ... --per-class 200 --link
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Class-balanced random subsample of an image folder.")
    ap.add_argument("--src", required=True, help="Source folder with one subfolder per class.")
    ap.add_argument("--out", default=None, help="Output folder (default: <src>_<N>_run<seed>).")
    ap.add_argument("--per-class", type=int, default=200, help="Images to sample per class (paper uses 200).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed. Change it for an independent run.")
    ap.add_argument("--link", action="store_true", help="Symlink instead of copy (saves disk).")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    out = Path(args.out).expanduser().resolve() if args.out \
        else src.parent / f"{src.name}_{args.per_class}_run{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    classes = sorted([d for d in src.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"No class subfolders found in {src}")

    total, short = 0, []
    print(f"Sampling {args.per_class}/class from {len(classes)} classes  (seed={args.seed})")
    print(f"  {src}  ->  {out}\n")

    for cls in classes:
        imgs = [p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(imgs)
        take = imgs[: args.per_class]
        if len(imgs) < args.per_class:
            short.append((cls.name, len(imgs)))

        dst_dir = out / cls.name
        dst_dir.mkdir(exist_ok=True)
        for p in take:
            dst = dst_dir / p.name
            if args.link:
                if not dst.exists():
                    dst.symlink_to(p)
            else:
                shutil.copy2(p, dst)
        total += len(take)
        print(f"  {cls.name:32s} {len(take):4d} / {len(imgs)}")

    print(f"\nDone: {total} images across {len(classes)} classes -> {out}")
    if short:
        print(f"Note: {len(short)} class(es) had fewer than {args.per_class} images "
              f"(took all): {', '.join(f'{n}({c})' for n, c in short)}")
    print(f"\nOpen in Bio Cluster: Browse -> {out}  ->  tick 'has labels'  ->  Run")


if __name__ == "__main__":
    main()
