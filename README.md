# BioCluster

[![bioRxiv preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.15.759340-bd2736.svg)](https://doi.org/10.64898/2026.10.15.759340)
[![DK-Moth dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23319153.svg)](https://doi.org/10.5281/zenodo.23319153)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)

Unsupervised species discovery and confidence-tiered annotation for biological
image collections.

Point it at a folder of images with no labels. It groups them by species using a
frozen biology foundation model, proposes a name for each group from a candidate
species list, and sorts every image into **auto-accept**, **review** or
**human-required** so that a specialist's time goes only where the model is
unsure.

It is a triage and pre-annotation tool, not a trained classifier. No model is
trained or fine-tuned on your data at any point.

The method and its evaluation are described in the preprint
[*BioCluster: Clustering and Zero-Shot Labeling of Ecological Images Using Foundation Models*](https://doi.org/10.64898/2026.10.15.759340)
(Ahmed, Bjerge, Høye and Pedersen, bioRxiv, 2026).

---

## Install

```bash
git clone https://github.com/sahmedAdnan/biocluster.git
cd biocluster
python -m venv .venv
. .venv/bin/activate                 # Linux/macOS
.venv\Scripts\activate               # Windows

pip install -e ".[gui]"              # desktop app + pipeline
pip install -e .                     # pipeline only, no GUI
```

Python 3.10 or 3.11 (the desktop app's theme package, `pyqtdarktheme`, does not
support 3.12 or newer yet). A CUDA GPU is strongly recommended for embedding
extraction; CPU works but is slow. On Windows, `pip` installs the CPU build of
PyTorch by default, so for GPU support install PyTorch first with the command from
[pytorch.org](https://pytorch.org/get-started/locally/). Pretrained weights download automatically on first use
(BioCLIP 2.5 is ~7 GB) and are cached by Hugging Face thereafter.

## Try it without any data

```bash
python examples/make_synthetic_dataset.py --out demo_data
biocluster-gui                       # open demo_data/
```

## DK-Moth dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23319153.svg)](https://doi.org/10.5281/zenodo.23319153)

**DK-Moth** is the camera-trap image dataset of Danish moths used in the [paper](https://doi.org/10.64898/2026.10.15.759340),
published on Zenodo under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It holds ten class-balanced subsamples of the same 248 species, the ten runs
the paper's means are computed over.

```text
DK-Moth/
├── DK-Moth_subsample248_classes_with_family.csv    species, genus, family, order
├── DK-moth_run01_seed42/                           2,310 images, one folder per species
│   ├── Abraxas sylvata/
│   └── ...
├── DK-moth_run02_seed43/
│   ...
└── DK-moth_run10_seed51/
```

| on Zenodo | contents |
|---|---|
| `DK-Moth_dataset.zip` (140 MB) | the folder tree above |
| `DK-Moth_subsample248_classes_with_family.csv` | the species list on its own, with taxonomy from GBIF |
| `DK-Moth_dataset_description_final.pdf` | how the dataset was collected, compiled and created |

**Using it.** Unzip, then point either app, or `dataset.root` in a
[`configs/paper/dkmoth_sub/`](configs/paper/dkmoth_sub) file, at one
`DK-moth_runNN_seedSS` folder with ground-truth labels on. To annotate with full
taxonomy, choose **Taxonomy CSV** in the app and pick the species list, or
uncomment `taxonomy_file:` in the config. Folder names carry only the genus, so
without the list the taxonomic-consistency signal has no family to check.

## Desktop application

```bash
biocluster-gui
```

Choose a folder, an encoder, a reduction and a clustering method, then run.
Five tabs: projection map, cluster distribution, evaluation metrics, image
verification (the review surface, worst-first by confidence), and run history.
Full guide in [docs/GUI.md](docs/GUI.md).

**Input layouts**

```
labelled                             unlabelled
root/                                root/
├── Vulpes_vulpes/                    ├── anything/
│   ├── 001.jpg                       │   └── 0001.jpg
│   └── 002.jpg                       ├── or_nested/by/date/
└── Capreolus_capreolus/              │   └── 0002.jpg
    └── 001.jpg                       └── 0003.jpg
```

Labelled mode needs one subfolder per species and reports accuracy against
those folder names. Unlabelled mode ignores structure, searches recursively, and
needs no labels at all. This is the normal mode for survey data. Accepted
formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

## Command line

Each stage is a separate command and writes to disk, so stages can be re-run
independently.

```bash
CFG=configs/paper/eumoth/backbone_bioclip2.5.yaml

biocluster-extract  --config $CFG     # 1. embeddings
biocluster-cluster  --config $CFG     # 2. reduce + cluster
biocluster-evaluate --config $CFG     # 3. metrics + figures
biocluster-annotate --config $CFG     # 4. name clusters + triage
biocluster-pipeline --config $CFG     # all of the above
```

`configs/` is organised so that each file reproduces one row of the paper:

```
configs/
├── demo_synthetic.yaml                     runs on generated data, no dataset needed
├── template.yaml                           documented starting point for new data
└── paper/<dataset>/
    ├── clustering_comparison.yaml          Table 2: six algorithms, BioCLIP 2
    └── backbone_<model>.yaml               Table 3: one backbone under HDBSCAN
```

Datasets are `eumoth`, `dkmoth_sub`, `cub200` and `hugoaves_sub`, and the six
backbones are `bioclip2.5`, `bioclip2`, `bioclip`, `dinov3`, `dinov2`, `clip`.
Each config states in its header which row it reproduces and what that row
reports, with the reduction and HDBSCAN settings taken from the logged run.
`dataset.root` is the only field you must set. See
[configs/README.md](configs/README.md) for the full traceability table, and for
the two rows whose published figures could not be traced exactly.

Reproduction is close but not bit-exact: EU-Moth / BioCLIP 2.5 re-run from its
config gives ARI 0.982 against the published 0.983, with 2 noise points instead
of 1, because embedding extraction is not bit-reproducible across batch sizes
and devices.

Four scripts cover the steps around a run that the configs cannot express:

| script | what it does |
|---|---|
| `scripts/prepare_hugo_bench.py` | downloads HUGO-Bench and writes it as an image folder |
| `scripts/subsample_balanced.py` | draws the class-balanced subsets the ten-run means are computed over |
| `scripts/pick_dbscan_eps.py` | derives DBSCAN's `eps` for one reduction, without using labels |
| `scripts/aggregate_results.py` | collapses `run_history.csv` into the per-configuration table |

## How it works

| stage | what happens |
|---|---|
| **Encode** | A frozen encoder (BioCLIP 2.5 / BioCLIP 2 / BioCLIP / CLIP / DINOv2 / DINOv3) maps each image to one vector. Cached on disk by model + file list. |
| **Reduce** | `none`, `pca`, `umap` or `pca+umap`. Applies to clustering only; text matching always uses the original space. |
| **Cluster** | k-means, agglomerative, spectral, GMM, DBSCAN or HDBSCAN. The density methods need no cluster count and may abstain. |
| **Name** | Each cluster centroid is matched to text embeddings of candidate species names. The candidate list is a vocabulary, not labels. |
| **Triage** | Four uncertainty signals (semantic margin, cluster membership, intra-cluster agreement, taxonomic consistency) combine into one score, split by two thresholds into three tiers. |

Details, including the encoder table and the default signal weights, in
[docs/PIPELINE.md](docs/PIPELINE.md).

## Two things to know before quoting a number

**Accuracy is computed over clustered images only.** Density-based methods leave
some images unassigned, and those are excluded from accuracy but included in the
tier counts. A configuration that abstains more therefore *looks* more accurate
while labelling less of the collection. Always read accuracy beside the retained
fraction, and never select a configuration on accuracy alone.

**The `human_required` tier is partly an abstention bucket.** Unclustered images
are named `Noise / Unknown` and routed there by construction, so on datasets with
high noise that tier's accuracy mostly reports the noise rate.

## Privacy and data handling

All computation is local. There is no cloud service and no third-party API. The
only outbound traffic is the one-time model-weight download and the optional
GBIF lookups, which send **species names only**, never images: one resolves a
name to its accepted taxonomy and match confidence, the other collects
photographs of a proposed species so a reviewer can check the name against
imagery from outside the collection. No model is trained or fine-tuned, so
nothing derived from a collection is retained in any model.

[docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md) specifies what an image
collection must provide, which optional metadata extends what can be measured,
and how extraction can be separated from analysis so that clustering runs on
embeddings alone.

## Where files are written

| path | contents |
|---|---|
| `~/BioCluster/feature_cache/` | cached embeddings |
| `~/BioCluster/run_history.csv` | one row per completed run |
| `results/` | metrics, figures and reports from CLI runs (git-ignored) |

## Development

```bash
pip install -e ".[gui,dev]"
pytest -q                            # synthetic only: no weights, no datasets
ruff check src tests scripts examples
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Citation

If you use BioCluster, please cite the paper:

```bibtex
@article{ahmed_2026_biocluster,
  author  = {Ahmed, Shakil and Bjerge, Kim and Høye, Toke T. and
             Pedersen, Christian Nørgaard Storm},
  title   = {{BioCluster}: Clustering and Zero-Shot Labeling of Ecological
             Images Using Foundation Models},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.15.759340},
  url     = {https://doi.org/10.64898/2026.10.15.759340}
}
```

If you use the DK-Moth dataset, please also cite:

```bibtex
@dataset{ahmed_2026_dkmoth,
  author    = {Ahmed, Shakil and Bjerge, Kim and Helsing-Nielsen, Flemming and
               Høye, Toke T. and Pedersen, Christian Nørgaard Storm},
  title     = {{DK-Moth}: a species-labelled camera-trap image dataset of Danish moths},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23319153},
  url       = {https://doi.org/10.5281/zenodo.23319153}
}
```

GitHub's **Cite this repository** button gives the software itself, from
[CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Built on [BioCLIP](https://imageomics.github.io/bioclip/) and [BioCLIP-2](https://imageomics.github.io/bioclip-2/)
(Imageomics Institute), [OpenCLIP](https://github.com/mlfoundations/open_clip),
DINOv2/v3 (Meta AI), [UMAP](https://umap-learn.readthedocs.io/) and
[HDBSCAN](https://hdbscan.readthedocs.io/). Taxonomic lookups use
[GBIF](https://www.gbif.org/).
