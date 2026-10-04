"""
Create a tiny synthetic image dataset so the pipeline can be exercised end to
end without any real data.

    python examples/make_synthetic_dataset.py --out demo_data --classes 6 --per-class 12
    biocluster-gui          # then point it at demo_data/

Each "species" is a coloured texture with per-image jitter. The point is to
verify wiring, not to produce biologically meaningful clusters.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for c in range(args.classes):
        base = rng.integers(40, 215, size=3)
        dirname = out / f"synthetic_species_{c:02d}"
        dirname.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            noise = rng.normal(0, 18, size=(args.size, args.size, 3))
            img = np.clip(base + noise, 0, 255).astype(np.uint8)
            # a per-class shape so the encoder has structure, not just colour
            yy, xx = np.mgrid[0:args.size, 0:args.size]
            mask = ((xx - args.size / 2) ** 2 + (yy - args.size / 2) ** 2
                    ) < (20 + 8 * c) ** 2
            img[mask] = (255 - base).astype(np.uint8)
            Image.fromarray(img).save(dirname / f"{i:03d}.jpg", quality=92)

    total = args.classes * args.per_class
    print(f"wrote {total} images in {args.classes} class folders -> {out}")
    print("labelled mode works on this layout; for unlabelled mode, flatten it.")


if __name__ == "__main__":
    main()
