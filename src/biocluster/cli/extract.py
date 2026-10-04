"""
CLI command: extract BioCLIP-2 embeddings and save them to disk.

Usage:
    biocluster-extract  --config configs/paper/eumoth/backbone_bioclip2.yaml

Output:
    <output_dir>/features.npz   -- embeddings, labels, class names
"""

import argparse
import os
import time

from biocluster.data.dataset import load_dataset
from biocluster.models.extractors import build_extractor

from biocluster.utils.io import ensure_dir, load_config, save_features, set_seed
from biocluster.utils.pretty import done, header, metrics_table, saved, subheader


def main(config_path: str = None) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Extract BioCLIP-2 image embeddings."
        )
        parser.add_argument("--config", required=True,
                            help="Path to YAML config file.")
        args        = parser.parse_args()
        config_path = args.config

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    header("STEP: Feature Extraction")

    # Step 1: Discover images
    subheader("Loading dataset")
    t0 = time.time()
    image_paths, labels, class_names = load_dataset(config)
    print("    Dataset    : " + config["dataset"]["name"])
    print("    Images     : " + str(len(image_paths)))
    print("    Classes    : " + str(len(class_names)))

    # Step 2: Load model
    subheader("Loading model")
    extractor = build_extractor(config)
    print("    Model      : " + config["model"]["name"])
    print("    Pretrained : " + config["model"]["pretrained"])

    # Step 3: Extract
    subheader("Extracting embeddings")
    features = extractor.extract(image_paths)

    elapsed = time.time() - t0
    metrics_table([
        {"label": "Feature shape",  "value": str(features.shape)},
        {"label": "Dim per image",  "value": features.shape[1]},
        {"label": "Time",           "value": "{:.1f}s".format(elapsed)},
    ], title="Extraction complete")

    # Step 4: Save
    out_path = os.path.join(output_dir, "features.npz")
    save_features(out_path, features, labels, image_paths, class_names)
    saved("features.npz", out_path)

    # Step 5: Record how these embeddings were produced. batch_size and
    # num_workers do not change the computation, but they do change its
    # floating-point result, because the GPU kernel chosen depends on the batch
    # shape. Without this, an embedding file cannot be traced to its settings.
    import json
    mcfg = config["model"]
    meta = {
        "model":        mcfg.get("name"),
        "pretrained":   mcfg.get("pretrained"),
        "batch_size":   mcfg.get("batch_size", 32),
        "num_workers":  mcfg.get("num_workers"),
        "device":       getattr(extractor, "device", mcfg.get("device", "")),
        "normalize_embeddings": mcfg.get("normalize_embeddings"),
        "global_pool":  mcfg.get("global_pool"),
        "img_size":     mcfg.get("img_size"),
        "seed":         config.get("seed", 42),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "n_images":     int(features.shape[0]),
        "dim":          int(features.shape[1]),
        "dataset_root": config["dataset"].get("root"),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = os.path.join(output_dir, "extraction_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if not isinstance(v, (int, float, str, type(None)))
                       else v) for k, v in meta.items()}, f, indent=2)
    saved("extraction_meta.json", meta_path)
    done(output_dir)


if __name__ == "__main__":
    main()
