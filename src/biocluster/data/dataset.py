"""
Image discovery and loading for supported dataset formats.

Supported formats:
  - imagefolder: class subfolders directly under the root directory
  - cub:         CUB-200-2011 with annotation files

To add a new format, see plugins.py beside this file.
"""

import os
import random
from collections import defaultdict
from typing import Tuple

from PIL import Image, UnidentifiedImageError

def load_imagefolder(root: str, extensions: list) -> Tuple[list, list, list]:
    """
    Load an ImageFolder-style dataset where each class is its own subfolder.

    Structure expected:
        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                ...

    Args:
        root:       Path to the dataset root (the folder containing class folders).
        extensions: List of valid image extensions, e.g. ['jpg', 'jpeg', 'png'].

    Returns:
        image_paths: Sorted list of full image file paths.
        labels:      Integer class ID for each image (aligned with image_paths).
        class_names: Sorted list of class name strings.
    """
    extensions = set(ext.lower().lstrip(".") for ext in extensions)

    # Discover and sort class folders for deterministic ordering
    class_names = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if not class_names:
        raise ValueError(f"No subdirectories found in: {root}")

    class_to_id = {name: i for i, name in enumerate(class_names)}

    image_paths = []
    labels      = []

    for class_name in class_names:
        class_dir = os.path.join(root, class_name)

        # Sort files inside each class folder too, for full determinism
        files = sorted(os.listdir(class_dir))

        for fname in files:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in extensions:
                full_path = os.path.join(class_dir, fname)
                image_paths.append(full_path)
                labels.append(class_to_id[class_name])

    if not image_paths:
        raise ValueError(
            f"No images found in {root} with extensions {extensions}. "
            "Check your dataset.root and image_extensions in the config."
        )

    print(f"[dataset] ImageFolder: {len(class_names)} classes, "
          f"{len(image_paths)} images found in {root}")

    return image_paths, labels, class_names


# -------------------------------------------------------
# CUB-200-2011 Loader
# -------------------------------------------------------

def load_cub(root: str, extensions: list) -> Tuple[list, list, list]:
    """
    Load the CUB-200-2011 dataset using its annotation files.

    Expected structure:
        root/
            images/
                001.Black_footed_Albatross/
                    ...
            images.txt       (image_id  relative_path)
            image_class_labels.txt (image_id  class_id)
            classes.txt      (class_id  class_name)

    Args:
        root:       Path to the CUB root directory.
        extensions: Not used for CUB (files are pre-listed), but kept for
                    API consistency.

    Returns:
        image_paths, labels, class_names
    """
    def read_txt(fname):
        with open(os.path.join(root, fname)) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        return lines

    # Load annotation files
    image_list  = read_txt("images.txt")             # [[id, rel_path], ...]
    label_list  = read_txt("image_class_labels.txt") # [[id, class_id], ...]
    class_list  = read_txt("classes.txt")            # [[class_id, class_name], ...]

    # Build lookup tables
    id_to_path     = {row[0]: row[1] for row in image_list}
    id_to_class_id = {row[0]: int(row[1]) - 1 for row in label_list}  # 0-indexed
    class_names    = [row[1] for row in sorted(class_list, key=lambda x: int(x[0]))]

    images_root = os.path.join(root, "images")

    # Build sorted lists (sort by image id for determinism)
    image_ids   = sorted(id_to_path.keys(), key=lambda x: int(x))
    image_paths = [os.path.join(images_root, id_to_path[i]) for i in image_ids]
    labels      = [id_to_class_id[i] for i in image_ids]

    print(f"[dataset] CUB-200-2011: {len(class_names)} classes, "
          f"{len(image_paths)} images found in {root}")

    return image_paths, labels, class_names


# -------------------------------------------------------
# Corrupted Image Filter
# -------------------------------------------------------

def filter_valid_images(image_paths: list, labels: list) -> Tuple[list, list]:
    """
    Remove corrupted or unreadable images from the lists.

    Uses a fast header-only check (PIL.Image.open without fully decoding).
    This is much faster than loading the full image.

    Returns:
        valid_paths, valid_labels  (same order, corrupted entries removed)
    """
    valid_paths  = []
    valid_labels = []
    skipped      = 0

    for path, label in zip(image_paths, labels):
        try:
            with Image.open(path) as img:
                img.verify()           # checks the header only
            valid_paths.append(path)
            valid_labels.append(label)
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1

    if skipped > 0:
        print(f"[dataset] Skipped {skipped} corrupted/unreadable image(s).")

    return valid_paths, valid_labels


# -------------------------------------------------------
# Subsampling
# -------------------------------------------------------

def subsample_dataset(image_paths: list, labels: list,
                      class_names: list, config: dict
                      ) -> Tuple[list, list, list]:
    """
    Subsample the dataset by selecting a random subset of classes and
    a fixed (or uncapped) number of images per class.

    Config keys (under dataset.subsample):
        enabled:              bool: must be True to activate
        n_classes:            int: how many classes to randomly select
        images_per_class:     int: images to sample per class
                                       use -1 to keep all images (uncapped)
        min_images_per_class: int: exclude classes with fewer images
        random_seed:          int: seed for reproducible subsampling

    Returns:
        subsampled (image_paths, labels, class_names) with contiguous
        label IDs starting from 0.
    """
    sub_cfg = config["dataset"]["subsample"]

    n_classes            = sub_cfg["n_classes"]
    images_per_class     = sub_cfg.get("images_per_class", 11)
    min_images_per_class = sub_cfg.get("min_images_per_class", images_per_class)
    seed                 = sub_cfg.get("random_seed", 42)
    uncapped             = (images_per_class == -1)

    # If uncapped, only filter by min_images_per_class
    if uncapped and min_images_per_class < 0:
        min_images_per_class = 1

    # Group image indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Filter: keep classes with enough images
    eligible = {cid: idxs for cid, idxs in class_indices.items()
                if len(idxs) >= min_images_per_class}

    print(f"[subsample] Eligible classes (>={min_images_per_class} images): "
          f"{len(eligible)} / {len(class_names)}")

    if len(eligible) < n_classes:
        print(f"[subsample] WARNING: only {len(eligible)} eligible classes, "
              f"requested {n_classes}. Using all {len(eligible)}.")
        n_classes = len(eligible)

    # Select classes
    rng = random.Random(seed)
    selected_cids = sorted(rng.sample(list(eligible.keys()), n_classes))

    # Sample images from each selected class
    new_paths  = []
    new_labels = []
    new_class_names = []

    for new_id, old_cid in enumerate(selected_cids):
        new_class_names.append(class_names[old_cid])
        idxs = eligible[old_cid]

        if uncapped:
            sampled = idxs
        else:
            sampled = rng.sample(idxs, min(images_per_class, len(idxs)))

        for idx in sorted(sampled):
            new_paths.append(image_paths[idx])
            new_labels.append(new_id)

    mode = "all" if uncapped else str(images_per_class)
    print(f"[subsample] Selected {n_classes} classes, {mode} images/class "
          f"-> {len(new_paths)} total images")

    return new_paths, new_labels, new_class_names


# -------------------------------------------------------
# Main Dispatcher
# -------------------------------------------------------

def load_dataset(config: dict) -> Tuple[list, list, list]:
    """
    Load a dataset according to the config's format key.

    Returns:
        image_paths, labels (int), class_names (str)
    """
    fmt        = config["dataset"]["format"]
    root       = config["dataset"]["root"]
    extensions = config["dataset"].get("image_extensions", ["jpg", "jpeg", "png"])

    if fmt == "imagefolder":
        paths, labels, class_names = load_imagefolder(root, extensions)
    elif fmt == "cub":
        paths, labels, class_names = load_cub(root, extensions)
    else:
        # Try to find a registered plugin
        from biocluster.data.plugins import get_plugin_loader
        loader = get_plugin_loader(fmt)
        paths, labels, class_names = loader(root, extensions, config)

    # Always filter corrupted files before returning
    paths, labels = filter_valid_images(paths, labels)

    # Subsample if configured
    sub_cfg = config.get("dataset", {}).get("subsample", {})
    if sub_cfg.get("enabled", False):
        paths, labels, class_names = subsample_dataset(
            paths, labels, class_names, config
        )

    return paths, labels, class_names
