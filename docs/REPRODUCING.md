# Reproducing the paper's results

There are two ways to reproduce the numbers in the
[paper](https://doi.org/10.64898/2026.10.15.759340):

| route | what you get | what you need |
|---|---|---|
| [A. Embeddings + command line](#a-exact-numbers-from-the-embeddings) | exactly the published numbers | our extracted embeddings; no GPU |
| [B. Desktop app](#b-the-desktop-app) | the published settings, run on embeddings extracted on your computer | the images and the model; a GPU helps |

> [!IMPORTANT]
> Extracted embeddings are not bit-identical across GPUs, drivers and batch
> sizes. The differences are tiny, but they can noticeably change the
> clustering downstream, so re-extracting the images will not always give the
> published numbers. We provide our extracted embeddings on
> [Google Drive](https://drive.google.com/drive/folders/1oHjNn059xV0iR30gyL6r-PbUkzpxcbMO?usp=sharing).
> Start from them (route A) to reproduce the paper exactly.

## What to download

- **DK-Moth** from Zenodo:
  [10.5281/zenodo.23319153](https://doi.org/10.5281/zenodo.23319153). The zip
  holds the ten subsamples (`DK-moth_run01_seed42` to `DK-moth_run10_seed51`,
  2,310 images in 248 species folders each) and the species list
  `DK-Moth_subsample248_classes_with_family.csv`.
- **Embeddings and run history** from
  [Google Drive](https://drive.google.com/drive/folders/1oHjNn059xV0iR30gyL6r-PbUkzpxcbMO?usp=sharing):
  one `.npz` per dataset, subsample and model, and `run_history.csv`, the log of
  every run behind the paper with its settings and scores.

## A. Exact numbers from the embeddings

Each file in `configs/paper/` reproduces one row of the paper. For one DK-Moth
subsample:

1. In `configs/paper/dkmoth_sub/backbone_bioclip2.5.yaml`, set `root` to the
   subsample folder, for example `.../DK-Moth/DK-moth_run01_seed42`.
2. Copy that subsample's BioCLIP 2.5 `.npz` from the Drive into the config's
   `output_dir` and rename it `features.npz`.
3. Run clustering and evaluation. This needs no model weights.

   ```bash
   CFG=configs/paper/dkmoth_sub/backbone_bioclip2.5.yaml
   biocluster-cluster  --config $CFG
   biocluster-evaluate --config $CFG
   ```

   In Windows PowerShell, set the variable with
   `$CFG = "configs/paper/dkmoth_sub/backbone_bioclip2.5.yaml"`.

4. For the annotation numbers, uncomment `taxonomy_file:` in the config and
   point it at `DK-Moth_subsample248_classes_with_family.csv`, then run the
   command below. This step loads the model's text encoder, so the weights are
   downloaded the first time.

   ```bash
   biocluster-annotate --config $CFG
   ```

Repeat for the ten subsamples and average. Each run overwrites `output_dir`,
so note the printed numbers, or give each subsample its own `output_dir`.

[configs/README.md](../configs/README.md) lists every config, the table row it
reproduces, and the two rows whose published figures could not be traced to a
single logged run.

## B. The desktop app

**DK-Moth.** The app starts with the DK-Moth default settings, so you only
choose the data:

1. **Browse Image Folder**: one subsample, for example `DK-moth_run01_seed42`.
2. Tick **Folder has Ground Truth Labels (Subfolders)**.
3. **Run Pipeline**.
4. In section 7, set **Candidate species from** to *Taxonomy CSV* and choose
   `DK-Moth_subsample248_classes_with_family.csv`. Leave the weights and
   thresholds as they are.
5. **Annotate Clusters**. The status line shows the tier counts and both
   accuracies.

Use the taxonomy CSV, not *Folder labels*. The paper's annotation numbers were
made with the CSV, and the folder names give slightly different results.

**Any other documented run.** Load the paper's run history into the app and
apply a run's settings:

1. Open the app once and close it, so that the folder `BioCluster` exists in
   your home folder (`C:\Users\<you>\BioCluster` on Windows, `~/BioCluster` on
   Linux and macOS).
2. Copy `run_history.csv` from the Drive into that folder. If a
   `run_history.csv` is already there, rename yours first, because the app
   reads only this file.
3. Keep a second copy of the downloaded file somewhere else as your reference.
   When you rerun a documented setting on a folder with the same name, the app
   replaces that run's row with your result.
4. Start the app and open **Run History**. Every documented run is listed.
   Filter by dataset, select a run, and click **Apply settings**. The model,
   reduction, clustering, their parameters and the seed are loaded into the
   panel. The image folder is not, because the stored paths are the authors'.
5. Choose your copy of the same dataset, tick the labels box, and run.

The run history does not store annotation settings. The paper used the app's
defaults: weights S 0.35, P 0.25, A 0.25, T 0.15, and thresholds 0.55 and 0.70.

## Expected DK-Moth numbers

BioCLIP 2.5 (ViT-H/14), UMAP (20 dimensions, `n_neighbors` 15, `min_dist`
0.15, cosine), HDBSCAN (`min_cluster_size` 4, `min_samples` 3, leaf), seed 42,
candidates from the taxonomy CSV, default weights and thresholds. Starting from
our embeddings (route A) gives exactly these numbers.

| subsample | ARI | NMI | Hungarian | clusters | noise % | auto | review | human | cluster acc | image acc |
|---|---|---|---|---|---|---|---|---|---|---|
| run01_seed42 | 0.841 | 0.962 | 0.889 | 242 | 10.82 | 1766 | 212 | 332 | 89.7 % (217/242) | 85.8 % (1768/2060) |
| run02_seed43 | 0.805 | 0.952 | 0.858 | 237 | 10.26 | 1722 | 238 | 350 | 89.9 % (213/237) | 82.6 % (1712/2073) |
| run03_seed44 | 0.841 | 0.960 | 0.876 | 246 | 11.65 | 1781 | 189 | 340 | 89.8 % (221/246) | 85.6 % (1747/2041) |
| run04_seed45 | 0.819 | 0.955 | 0.873 | 242 | 11.26 | 1739 | 216 | 355 | 86.8 % (210/242) | 83.1 % (1703/2050) |
| run05_seed46 | 0.818 | 0.956 | 0.871 | 234 | 10.00 | 1753 | 215 | 342 | 89.7 % (210/234) | 83.4 % (1734/2079) |
| run06_seed47 | 0.793 | 0.953 | 0.855 | 240 | 10.26 | 1724 | 233 | 353 | 90.0 % (216/240) | 83.3 % (1726/2073) |
| run07_seed48 | 0.787 | 0.950 | 0.846 | 239 | 9.22 | 1721 | 277 | 312 | 87.4 % (209/239) | 81.4 % (1708/2097) |
| run08_seed49 | 0.826 | 0.957 | 0.877 | 246 | 11.56 | 1752 | 196 | 362 | 92.3 % (227/246) | 86.4 % (1765/2043) |
| run09_seed50 | 0.810 | 0.953 | 0.860 | 244 | 12.12 | 1763 | 190 | 357 | 90.2 % (220/244) | 83.7 % (1699/2030) |
| run10_seed51 | 0.818 | 0.954 | 0.858 | 242 | 10.69 | 1747 | 219 | 344 | 89.7 % (217/242) | 83.8 % (1729/2063) |
| **mean** | **0.816** | **0.955** | **0.866** | **241.2** | **10.78** | **1746.8** | **218.5** | **344.7** | **89.5 %** | **83.9 %** |

- **ARI, NMI and Hungarian** are computed on clustered images only, leaving
  noise out.
- **auto, review and human** count all 2,310 images, noise included.
- **Image acc** counts clustered images whose proposed species is their true
  species. **Cluster acc** counts clusters whose name matches the most common
  true species in them.

The mean ARI, clusters and noise are the DK-Moth row of the paper's backbone
table (0.816, 241, 10.8 %). The mean image accuracy is the 84 % in the
abstract.

## Other things that change a run

- **The seed.** UMAP and the seeded clustering methods depend on it. Keep it
  at 42.
- **The 2-D map.** The projection map is redrawn on every run and its layout
  can change. Clusters and scores do not.
