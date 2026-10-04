# Configurations

Every file under `paper/` reproduces a specific row of the manuscript, with the
parameters taken from the logged run rather than re-derived. Set
`dataset.root` and run:

```bash
biocluster-extract  --config configs/paper/<dataset>/<file>.yaml
biocluster-cluster  --config configs/paper/<dataset>/<file>.yaml
biocluster-evaluate --config configs/paper/<dataset>/<file>.yaml
biocluster-annotate --config configs/paper/<dataset>/<file>.yaml
```

| file | purpose |
|---|---|
| `demo_synthetic.yaml` | runs end to end on generated data, no real dataset needed |
| `template.yaml` | documented starting point for a new dataset |
| `paper/<dataset>/clustering_comparison.yaml` | Table 2: six algorithms on BioCLIP 2 (ViT-H/14) |
| `paper/<dataset>/backbone_<model>.yaml` | Table 3: one backbone under HDBSCAN |

Datasets: `eumoth`, `dkmoth_sub`, `cub200`, `hugoaves_sub`.

## Table 3: backbone comparison (HDBSCAN)

Reduction, HDBSCAN setting and the reported figures behind each file:

| dataset | backbone | reduction | mcs / ms | ARI | clusters | noise % |
|---|---|---|---|---|---|---|
| EU-Moth | BioCLIP 2 (ViT-H/14) | UMAP-20 | 5 / 5 | 0.983 | 200 | 0.05 |
| EU-Moth | BioCLIP 2 (ViT-L/14) | UMAP-20 | 5 / 5 | 0.978 | 200 | 0.09 |
| EU-Moth | BioCLIP | UMAP-10 | 3 / 3 | 0.895 | 194 | 2.9 |
| EU-Moth | DINOv3 | UMAP-20 | 4 / 2 | 0.525 | 181 | 13.92 |
| EU-Moth | DINOv2 | UMAP-20 | 3 / 2 | 0.458 ⚠ | 219 | 20.5 |
| EU-Moth | CLIP | UMAP-15 | 3 / 3 | 0.425 | 162 | 33.15 |
| DK-Moth (sub) | BioCLIP 2 (ViT-H/14) | UMAP-20 | 4 / 3 | 0.816 | 241 | 10.8 |
| DK-Moth (sub) | BioCLIP 2 (ViT-L/14) | UMAP-20 | 4 / 2 | 0.727 | 248 | 14.5 |
| DK-Moth (sub) | BioCLIP | UMAP-10 | 4 / 2 | 0.482 | 252 | 20.4 |
| DK-Moth (sub) | DINOv3 | UMAP-20 | 4 / 2 | 0.524 | 231 | 25.4 |
| DK-Moth (sub) | DINOv2 | UMAP-20 | 4 / 2 | 0.434 | 230 | 26.4 |
| DK-Moth (sub) | CLIP | UMAP-15 | 4 / 2 | 0.273 | 216 | 31.6 |
| CUB-200-2011 | BioCLIP 2 (ViT-H/14) | PCA-150+UMAP-5 | 25 / 3 | 0.884 | 202 | 1.01 |
| CUB-200-2011 | BioCLIP 2 (ViT-L/14) | PCA-150+UMAP-5 | 20 / 20 | 0.882 | 200 | 1.9 |
| CUB-200-2011 | BioCLIP | PCA-100+UMAP-5 | 10 / 10 | 0.78 | 190 | 6.17 |
| CUB-200-2011 | DINOv3 | UMAP-5 | 15 / 4 | 0.716 | 198 | 4.88 |
| CUB-200-2011 | DINOv2 | UMAP-5 | 10 / 5 | 0.715 | 216 | 4.98 |
| CUB-200-2011 | CLIP | UMAP-5 | 10 / 10 | 0.6 | 161 | 17.74 |
| HUGO-Bench Aves (sub) | BioCLIP 2 (ViT-H/14) | UMAP-20 | 70 / 7 | 0.853 | 32 | 15.4 |
| HUGO-Bench Aves (sub) | BioCLIP 2 (ViT-L/14) | UMAP-20 | 70 / 7 | 0.851 | 31 | 20.5 |
| HUGO-Bench Aves (sub) | BioCLIP | UMAP-10 | 70 / 7 | 0.614 | 27 | 34.8 |
| HUGO-Bench Aves (sub) | DINOv3 | UMAP-10 | 70 / 7 | 0.92 ⚠ | 31 | 2.6 |
| HUGO-Bench Aves (sub) | DINOv2 | UMAP-5 | 70 / 7 | 0.843 | 32 | 4.7 |
| HUGO-Bench Aves (sub) | CLIP | UMAP-10 | 70 / 7 | 0.676 | 28 | 34.7 |

⚠ marks a row whose published figures could not be traced exactly to a single
logged run. The config says what the discrepancy is; both cases are small and
documented rather than silently reconciled:

- **EU-Moth / DINOv2**: the published row takes its cluster count from the
  `min_dist 0.20` sweep run and its noise fraction from the `min_dist 0.15`
  run. The config uses 0.20, which matches the reported 219 clusters, and so
  yields 21.09 % noise rather than the printed 20.5 %.
- **HUGO-Bench Aves / DINOv3**: the only ten-run group in the history gives
  ARI 0.9243 and 3.66 % noise against the published 0.920 and 2.6 %. The cache
  holds two DINOv3 extractions per subsample (default pooling and CLS token)
  and the history does not record which produced the published row; the config
  sets `global_pool: token`, matching the paper's footnote.

## Conventions these configs encode

- **Thresholds are absolute**, 0.55 and 0.70, not percentiles, so a strong
  clustering flags few images and a weak one flags many.
- **Uncertainty weights** are 0.35 / 0.25 / 0.25 / 0.15 over the semantic,
  spatial, cluster-agreement and taxonomic signals.
- **Cluster selection** is `eom` on the curated benchmarks and `leaf` on the
  camera-trap subsamples, following each embedding's separability.
- **DINOv2 and DINOv3 use `global_pool: token`** (the CLS token), per the
  manuscript footnote. They have no text encoder, so they cluster but cannot
  name clusters; their configs omit the annotation block.
- **DBSCAN's `eps` is only valid for the reduction it sits beside.** It is an
  absolute radius, so changing the reduction requires re-deriving it with
  `scripts/pick_dbscan_eps.py`.
- **The two subsampled datasets are reported as ten-run means.** Each config
  runs a single subsample; point `root` at each in turn to reproduce the mean.
