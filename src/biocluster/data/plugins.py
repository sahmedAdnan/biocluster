"""
Plugin registry for custom dataset formats.

If your dataset does not fit the 'imagefolder' or 'cub' formats,
register a loader function here and reference its key in your YAML config.

How to add a new plugin:
  1. Write a function with the signature:
         def my_loader(root, extensions, config) -> (paths, labels, class_names)
  2. Register it by adding an entry to PLUGIN_REGISTRY below.
  3. Set format: my_key in your YAML config.
"""

# -------------------------------------------------------
# Registry
# -------------------------------------------------------

PLUGIN_REGISTRY = {}


def register_plugin(key: str):
    """Decorator to register a dataset loader function under a given key."""
    def decorator(fn):
        PLUGIN_REGISTRY[key] = fn
        return fn
    return decorator


def get_plugin_loader(key: str):
    """Retrieve a registered loader function by key."""
    if key not in PLUGIN_REGISTRY:
        raise ValueError(
            f"Unknown dataset format: '{key}'. "
            f"Available formats: imagefolder, cub, "
            f"{list(PLUGIN_REGISTRY.keys())}"
        )
    return PLUGIN_REGISTRY[key]


# -------------------------------------------------------
# Example Plugin: iNaturalist-style annotation file
# -------------------------------------------------------

@register_plugin("inaturalist")
def load_inaturalist(root: str, extensions: list, config: dict):
    """
    Example plugin for an iNaturalist-style dataset with a JSON annotation file.

    Config expects:
        dataset:
          format: inaturalist
          annotation_file: path/to/annotations.json

    The JSON should have:
        {
          "images":  [{"id": 1, "file_name": "...", "category_id": 5}, ...],
          "categories": [{"id": 5, "name": "Panthera leo"}, ...]
        }
    """
    import json
    import os

    ann_file = config["dataset"].get("annotation_file")
    if ann_file is None:
        raise ValueError("iNaturalist plugin requires 'annotation_file' in config.")

    with open(ann_file) as f:
        data = json.load(f)

    # Build class lookup
    categories  = sorted(data["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    cat_id_map  = {c["id"]: i for i, c in enumerate(categories)}

    # Build image lists (sort by id for determinism)
    images = sorted(data["images"], key=lambda x: x["id"])

    image_paths = [os.path.join(root, img["file_name"]) for img in images]
    labels      = [cat_id_map[img["category_id"]] for img in images]

    print(f"[plugin/inaturalist] {len(class_names)} classes, "
          f"{len(image_paths)} images.")

    return image_paths, labels, class_names
