"""
BioCluster: unsupervised species discovery and annotation for biological images.

Pipeline: frozen image encoder -> dimensionality reduction -> clustering ->
zero-shot species naming -> four-signal uncertainty -> confidence tiers.

Public submodules
-----------------
    data        dataset discovery and loading
    models      embedding extractors (BioCLIP 2/2.5, BioCLIP, CLIP, DINOv2/v3)
    reduction   none / PCA / UMAP / PCA+UMAP
    clustering  k-means, agglomerative, spectral, GMM, DBSCAN, HDBSCAN
    evaluation  ARI, NMI, Hungarian accuracy, confusion analysis
    annotation  cluster naming, uncertainty signals, confidence tiers, review
    app         desktop GUI
    cli         command-line stages
"""

import os as _os

# Deterministic cuBLAS kernels. This has to be set before CUDA initialises, and
# importing any biocluster submodule runs this file first, so here is the only
# place that holds for the GUI, the CLI and notebooks alike. Without it,
# set_seed()'s torch.use_deterministic_algorithms(..., warn_only=True) degrades
# to fast non-deterministic kernels without raising, and extraction stops being
# reproducible. setdefault, so an explicit value from the environment wins.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
__all__ = ["data", "models", "reduction", "clustering", "evaluation",
           "annotation", "visualization", "utils"]
