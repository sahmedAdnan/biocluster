# Desktop app guide

The desktop app runs the full BioCluster pipeline without a config file:
feature extraction, dimensionality reduction, clustering, zero-shot
annotation, confidence tiers and expert review. This guide covers installing,
running, reviewing and exporting.

The app starts with the DK-Moth default settings, the ones used for DK-Moth in
the paper ([section 3](#3-the-settings-panel)). To reproduce the paper's
numbers, see [REPRODUCING.md](REPRODUCING.md).

- [1. Install and start](#1-install-and-start)
- [2. A first run on demo data](#2-a-first-run-on-demo-data)
- [3. The settings panel](#3-the-settings-panel)
- [4. The result tabs](#4-the-result-tabs)
- [5. Reviewing the annotations](#5-reviewing-the-annotations)
- [6. The web report](#6-the-web-report)
- [7. Files and exports](#7-files-and-exports)
- [8. Troubleshooting](#8-troubleshooting)

## 1. Install and start

```bash
git clone https://github.com/sahmedAdnan/biocluster.git
cd biocluster
python -m venv .venv
.venv\Scripts\activate               # Windows
. .venv/bin/activate                 # Linux/macOS

pip install -e ".[gui]"
biocluster-gui
```

- **Python 3.10 or 3.11.** The theme package does not install on 3.12 or newer.
- **GPU.** On Windows, `pip` installs the CPU build of PyTorch. If you have an
  NVIDIA GPU, install PyTorch first with the command from
  [pytorch.org](https://pytorch.org/get-started/locally/), then run the
  `pip install` line above. The app works on CPU, but extraction is slow.
- **First run.** The first time a model is used, its weights download from
  Hugging Face (BioCLIP 2.5 is about 7 GB) and are cached after that.
- If `biocluster-gui` is not found, the virtual environment is not active.
  `python -m biocluster.app.main` starts the same app.

## 2. A first run on demo data

```bash
python examples/make_synthetic_dataset.py --out demo_data
biocluster-gui
```

1. **Browse Image Folder** and choose `demo_data`.
2. Tick **Folder has Ground Truth Labels (Subfolders)**.
3. Click **Run Pipeline**, keeping the DK-Moth default settings.
4. When the run finishes, click **Annotate Clusters** in section 7.

The demo images are coloured textures, so the clusters mean nothing
biologically. The run only checks that everything is installed and working.

## 3. The settings panel

The panel on the left is numbered in the order the steps run. The app starts
with the DK-Moth default settings, the ones used for DK-Moth in the paper:

| setting | default |
|---|---|
| Model | BioCLIP 2.5 Huge (ViT-H/14) |
| Reduction | UMAP, `n_components` 20, `n_neighbors` 15, `min_dist` 0.15, `metric` cosine |
| Clustering | HDBSCAN, `min_cluster_size` 4, `min_samples` 3, `cluster_selection_method` leaf |
| Random seed | 42 |
| Uncertainty weights | S 0.35, P 0.25, A 0.25, T 0.15 |
| Tier thresholds | review 0.55, human 0.70 |

These suit camera-trap insect images with a few to a few dozen images per
species. Hover over any setting to see what changing it does.

### 1. Data Input

**Browse Image Folder** picks the collection. Accepted formats are `.jpg`,
`.jpeg`, `.png` and `.webp`.

Tick **Folder has Ground Truth Labels (Subfolders)** only when every subfolder
is one species, as in a DK-Moth subsample. The folder names then count as the
true labels, the metrics tab reports accuracy, and the folder names can serve
as the candidate list.

```
DK-moth_run01_seed42/                          labelled: one folder per species
├── Abraxas sylvata/
│   ├── Abraxas sylvata_SS2_197_1_0_3.JPG
│   └── ...
├── Acasis viretata/
│   └── ...
├── ...                                        248 species folders
└── Ypsolopha parenthesella/
```

Leave it unticked for survey data. Images are then found in all subfolders,
whatever their structure, and no labels are needed.

```
root/                                          unlabelled: any structure
├── anything/
│   └── 0001.jpg
├── or_nested/by/date/
│   └── 0002.jpg
└── 0003.jpg
```

### 2. Feature Extraction

| model in the app | can name clusters | notes |
|---|---|---|
| BioCLIP 2.5 Huge (ViT-H/14) | yes | default, and the best encoder in the paper |
| BioCLIP2 | yes | |
| BioCLIP (Original) | yes | |
| CLIP (ViT-L/14) | yes | general-purpose, not trained on biology |
| DINOv2 (ViT-L/14) | no | vision only: clusters, but cannot annotate |
| DINOv3 (ViT-H+/16) | no | vision only: clusters, but cannot annotate |

**Pooling** appears only for the DINO models. Use *CLS token* when the animal
is small in the frame and *Mean patches* when images are already cropped to it.

Extraction is the slow step. BioCLIP 2.5 runs at about 16 images per second on
a 6 GB laptop GPU (RTX 4050), so about 11 minutes per 10,000 images. The
embeddings are cached per model and file list, so the next run on the same
folder starts straight at clustering. Adding, removing or editing an image
starts a fresh extraction.

The GPU, driver and batch size change the extracted embeddings very slightly,
and that can noticeably change the clustering. To reproduce the paper, start
from our extracted embeddings, as described in [REPRODUCING.md](REPRODUCING.md).

### 3. Dimensionality Reduction

Reduction only affects clustering. Naming always uses the full embeddings.

| method | when to use it |
|---|---|
| `umap` (default) | most collections; 20 dimensions worked for both moth datasets |
| `pca+umap` | very large or noisy embeddings; PCA first, then UMAP (used for CUB-200) |
| `pca` | fast and deterministic, but usually separates species less well |
| `none` | cluster the full embedding |

Raise `n_neighbors` when each species has many images. HUGO-Bench, with about
200 images per species, used 70.

### 4. Clustering

**HDBSCAN** (the default) finds the number of clusters itself. It can leave
images it cannot place unassigned, which the app calls *noise*. It also
reports how firmly each image belongs to its cluster, which feeds the spatial
uncertainty signal. With other methods the app uses the silhouette score
instead.

- `min_cluster_size`: roughly the smallest number of images you expect for a
  species. DK-Moth has up to 10 images per species and uses 4. HUGO-Bench uses
  70 and CUB-200 uses 25.
- `min_samples`: higher values are stricter and leave more images as noise.
- `cluster_selection_method`: `leaf` gives many fine clusters (used for the
  camera-trap datasets); `eom` gives fewer, larger ones (used for the curated
  benchmarks).

k-means, agglomerative, spectral and GMM need `n_clusters`, the number of
species you expect. They place every image, so nothing is left as noise. DBSCAN
needs `eps`, a distance that only holds for one reduction setting;
`scripts/pick_dbscan_eps.py` derives it without labels.

### 5. Hardware Configuration

- **Device**: your GPU if one was found, otherwise CPU.
- **Batch Size**: 64 on GPU and 16 on CPU by default. Lower it if you run out
  of memory.
- **Num Workers**: leave at 0 on Windows.

### 6. Execution

**Random Seed** makes UMAP and the seeded clustering methods repeatable. The
line under it shows which steps it affects. **Run Pipeline** turns into
**Cancel** while a run is going. When the run finishes, **Export Results
(CSV)** appears (see [section 7](#7-files-and-exports)).

### 7. Annotation (after clustering)

Run the pipeline first, then click **Annotate Clusters**.

**Candidate species from** sets the list of names a cluster can be given. A
species that is not on the list can never be proposed.

- *Folder labels* uses the folder names. This needs the labels box from
  section 1 ticked before the run.
- *Taxonomy CSV* reads a file with the columns `class_name, genus, family,
  order`. With a family column, each prompt carries the full taxonomy and the
  taxonomic signal can check family membership. For DK-Moth, use
  `DK-Moth_subsample248_classes_with_family.csv` from the dataset zip.

**Web report: GBIF reference photos** only affects the web report
([section 6](#6-the-web-report)).

**Uncertainty weights and thresholds.** Every image gets four uncertainty
signals between 0 and 1, where higher means less certain:

| signal | what it measures |
|---|---|
| **S** semantic | how close the runner-up species is to the top choice |
| **P** spatial | how weakly the image belongs to its cluster |
| **A** agreement | whether the image on its own agrees with its cluster's name |
| **T** taxonomic | whether the top choices fall in the same family |

The **Uncertainty signal weights** set how much each signal counts, and their
weighted average is the combined uncertainty. Below the review threshold an
image is `auto_accept`, between the two thresholds it is `review`, and at or
above the human threshold it is `human_required`. Noise images are named
`Noise / Unknown` and always go to `human_required`.

When annotation finishes, the status line reads, for example:

```
Done: 1766 auto · 212 review · 332 human  ·  img acc 85.8% (1768/2060)  ·  cluster acc 89.7% (217/242)
```

The two accuracies appear only with labelled folders. *img acc* counts
clustered images only, leaving noise out. *cluster acc* counts a cluster as
correct when its name matches the most common true species in it.

Annotation uses the model chosen in the **Model** box. Keep it on the model you
ran the pipeline with. Only CLIP-type models can annotate.

## 4. The result tabs

| tab | what it shows |
|---|---|
| **Projection Map** | a 2-D map of the images, coloured by cluster; hover to see an image |
| **Cluster Distribution** | cluster sizes, and how much of the collection each holds |
| **Evaluation Metrics** | ARI, NMI and Hungarian accuracy with labelled folders; silhouette, Calinski-Harabasz and Davies-Bouldin without |
| **Image Verification** | the review screen ([section 5](#5-reviewing-the-annotations)) |
| **Run History** | every completed run with its settings and scores |

The 2-D map is redrawn on every run and its layout can change. The clusters
and scores do not change for the same settings and seed.

The labelled metrics leave noise images out, so read them beside the noise
percentage. A run that leaves more images unassigned is graded on an easier
subset.

**Run History** stores one row per run in `~/BioCluster/run_history.csv`.
Filter by dataset, and tick **Group by config** to see the mean and standard
deviation of runs with the same settings on the same folder. Select a run to
see all its settings. **Apply settings** loads them back into the panel, and
**Delete** removes the run from the history.

Every run behind the paper is in `run_history.csv` on our
[Google Drive](https://drive.google.com/drive/folders/1oHjNn059xV0iR30gyL6r-PbUkzpxcbMO?usp=sharing).
Copy it into your `BioCluster` folder to browse those runs and apply their
settings. [REPRODUCING.md](REPRODUCING.md#b-the-desktop-app) explains how,
without losing your own history.

## 5. Reviewing the annotations

**Image Verification** opens by itself when annotation finishes.

1. **Filter** by tier. Start with *Human Required*, then *Review*. Spot-check
   *Auto-accepted*. *Noise* shows the images no cluster took.
2. **Select Cluster to Verify** lists the clusters with the least certain
   first. Above the images you see the proposed name, the runner-up, and
   reference images of both species from your collection, so you can compare.
3. **Correct a whole cluster** with the buttons above it:
   - *Confirm cluster (AI)* accepts the proposed name.
   - *Confirm runner-up* gives every image the second choice.
   - Type or pick a name, then *Apply label to cluster*.
   - *All noise* or *All junk* marks every image in the cluster.
4. **Correct single images** on their cards. Each card shows its confidence,
   tier and S/P/A/T values, plus *✓ top-1*, *→ runner-up*, a name box, *noise*,
   *junk* and *reset*. Click a thumbnail to enlarge it.
5. **Undo** and **Redo** step through your changes.

Corrections are saved as you work to `~/BioCluster/autosave_<id>.json`. The
next time you annotate the same folder, the app offers to restore them. **Save
session** and **Load session** write and read a named JSON file, for example
to continue on another computer.

**Export Decisions (CSV)** writes the checked result: one row per image with
`file_path`, `final_label` and `status` (`species`, `noise` or `junk`).

## 6. The web report

**Generate Web Report** writes a single `review.html` file for a reviewer who
does not have the app installed. It shows each cluster, least certain first,
with its proposed name, its runner-up, and reference photographs of both.

Where those photographs come from depends on the **Web report: GBIF reference
photos** tick-box:

- **Unticked:** images from the collection's own species folders. This needs
  labelled folders and shows nothing for an unlabelled collection. It is also
  not an independent check, because the proposed name came from those images.
- **Ticked:** photographs of the proposed species from GBIF, which checks the
  name against images from outside the collection. Only adults are shown,
  each captioned with its recorded life stage, because most GBIF photos of
  moths are of caterpillars. Only the names shown in the report are looked up,
  behind a progress window you can cancel. Results are cached in
  `~/BioCluster/gbif_cache/`, so a second report needs no new lookups.

GBIF photographs are contributed by the public and are not necessarily
checked by experts, so they support a visual comparison, not an
identification. The report stores links rather than the photos, so it needs
an internet connection to show them. Everything else in it works offline.

The report embeds a thumbnail of every image, about 12 KB each, so it grows by
roughly 12 MB per 1,000 images.

## 7. Files and exports

What you export:

| button | file | contents |
|---|---|---|
| Export Results (CSV) | `cluster_results.csv` | per image: `path`, `filename`, `cluster_id`, `cluster_label` (`Noise` for unassigned), `true_class` with labelled folders, and after annotation `ai_annotation`, `ai_second_choice`, `margin`, `combined_uncertainty`, `confidence_tier` |
| Export Decisions (CSV) | `decisions.csv` | per image: `file_path`, `final_label`, `status` |
| Save session | `review_session.json` | your corrections, to load again later |
| Generate Web Report | `review.html` | the self-contained review page |

What the app keeps on its own:

| path | contents |
|---|---|
| `~/BioCluster/feature_cache/` | cached embeddings, keyed by model and file list |
| `~/BioCluster/run_history.csv` | one row per completed run: settings and metrics |
| `~/BioCluster/autosave_<id>.json` | corrections in progress, one file per folder |
| `~/BioCluster/gbif_cache/` | cached GBIF name matches and photo links |
| `~/BioCluster/startup_error.log` | crash log, written only when the app fails to start |

`~` is your home folder, for example `C:\Users\<you>` on Windows.

Nothing is uploaded. The app only goes online to download model weights once,
and for the optional GBIF lookups, which send species names only, never
images.

## 8. Troubleshooting

| problem | what to do |
|---|---|
| `biocluster-gui` is not recognised | Activate the virtual environment, or start with `python -m biocluster.app.main`. |
| `No module named 'qdarktheme'` | `pip install pyqtdarktheme==2.1.0` (needs Python 3.10 or 3.11). |
| Device shows only CPU on a computer with an NVIDIA GPU | PyTorch is the CPU build. Install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/). |
| Out of GPU memory | Lower **Batch Size**, close other programs that use the GPU, or choose a smaller model. |
| Extraction gets slower after several runs in one session | Restart the app. The embeddings already extracted stay cached. |
| **Annotate Clusters** is greyed out | Run the pipeline first. |
| "Annotation unavailable" | DINOv2 and DINOv3 cannot name clusters. Re-run with a BioCLIP or CLIP model. |
| "No labels available" | *Folder labels* needs the labels box ticked before the run. Tick it and re-run, or choose a taxonomy CSV. |
| The app does not start | Read `~/BioCluster/startup_error.log`. |
