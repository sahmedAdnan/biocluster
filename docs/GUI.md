# The desktop application

```bash
pip install -e ".[gui]"
biocluster-gui
```

A PySide6 application that runs the whole pipeline without writing a config
file. Point it at a folder of images, pick an encoder, reduction and clustering
method, and run.

## Tabs

| tab | what it is for |
|---|---|
| **Projection Map** | 2-D projection of the embedding, coloured by cluster; hover to see the image |
| **Cluster Distribution** | cluster sizes, and how much of the collection each one holds |
| **Evaluation Metrics** | ARI, NMI, Hungarian accuracy when the folders carry labels; silhouette, Calinski-Harabasz and Davies-Bouldin when they do not |
| **Image Verification** | the review surface: clusters ordered worst-first by confidence tier, with the proposed species name, so a specialist corrects the uncertain ones first |
| **Run History** | every completed run with its settings and scores, appended to `~/BioCluster/run_history.csv`, so a configuration can be reproduced or re-applied |

## Two input modes

**Labelled**: one subfolder per species (`root/species_name/img.jpg`). Folder
names become ground truth, and the metrics tab reports accuracy.

**Unlabelled**: any folder structure; images are found recursively. This is the
normal mode for real survey data. Ground-truth metrics are unavailable, so
cluster quality is judged by silhouette and by review.

## The web report

After annotation, **Generate Web Report** writes a self-contained
`review.html` for a reviewer who does not have the application installed. It
shows each cluster worst-first with its proposed name, its runner-up, and
reference photographs for both.

Where those photographs come from depends on the **GBIF reference photos**
tick-box in the annotation panel:

- unticked: images from the collection's own class directories. This needs
  labelled folders, and shows nothing on an unlabelled collection. It also is
  not an independent check of a proposed name, because the proposal came from
  those images.
- ticked: photographs of the proposed species fetched from GBIF, which is the
  comparison that checks a name against imagery from outside the collection.
  Adults only, each captioned with its recorded life stage, because GBIF's
  media for moths is mostly caterpillars and an unlabelled caterpillar photo
  is not something a reviewer can check an adult against. Only the names on
  screen are fetched, which is each cluster's top choice and runner-up, behind
  a progress dialog that can be cancelled. Results are cached in
  `~/BioCluster/gbif_cache/`, so a second report costs nothing.

GBIF occurrence photographs are contributor-submitted and are not necessarily
expert-verified, so they support a visual comparison rather than a
determination. Because only URLs are stored, the report needs a connection
when it is opened to show those panels; everything else in it works offline.

## What is stored where

| path | contents |
|---|---|
| `~/BioCluster/feature_cache/` | cached embeddings, keyed by model + file list |
| `~/BioCluster/run_history.csv` | one row per completed run: settings and metrics |
| `~/BioCluster/autosave_<hash>.json` | in-progress expert corrections for a dataset |
| `~/BioCluster/gbif_cache/` | cached GBIF name matches and reference photo URLs |
| `~/BioCluster/startup_error.log` | crash log, written only on a failed launch |

Nothing is uploaded. The only network access is the one-time download of
pretrained weights from Hugging Face, plus the optional GBIF taxonomy and
reference-photo lookups, which send species names only, never images.
