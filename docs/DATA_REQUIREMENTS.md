# Data specification

What the pipeline requires of an image collection, and which optional fields
enable which analyses. The requirements follow from the implementation: the
loaders in `biocluster/data/dataset.py`, the extractors in
`biocluster/models/extractors.py`, and the annotation stage in
`biocluster/annotation/`.

## 1. Required input

A directory of image files. Nothing else is required to cluster a collection.

| Property | Requirement |
|---|---|
| File formats | `.jpg`, `.jpeg`, `.png`, `.webp` |
| Directory layout | Arbitrary. The unlabelled loader searches subdirectories recursively and ignores their structure |
| File names | Arbitrary. No information is parsed from a file name |
| Species labels | Not required. The clustering and naming stages are unsupervised |
| Colour | RGB or greyscale. Infrared night frames are accepted, and behave differently from daylight frames (Section 5) |

Unreadable or truncated files are detected by a header check and excluded
before extraction, so a collection need not be cleaned in advance.

### Two layouts

Labelled mode expects one subdirectory per class, and raises an error when the
root contains no subdirectories:

```
root/
├── Capreolus_capreolus/
│   ├── 0001.jpg
│   └── 0002.jpg
└── Vulpes_vulpes/
    └── 0001.jpg
```

Directory names become the ground-truth labels and enable ARI, NMI and
Hungarian accuracy. Unlabelled mode places no constraint on layout:

```
root/
├── site_a/2024-05-01/IMG_0001.jpg
└── site_b/IMG_0002.jpg
```

Without labels, cluster quality is assessed by the silhouette score and by
expert review of the generated report.

## 2. Collection size

| Quantity | Guidance |
|---|---|
| Total images | Useful results from about 2,000 upward |
| Images per species | About 10 or more |
| Species represented by 1 or 2 images | Usually left unassigned rather than clustered |

HDBSCAN forms a cluster only from at least `min_cluster_size` points, so the
smallest species in a collection sets a floor on that parameter. The datasets
used for evaluation span the usable range: about 9 images per species for the
moth collections with `min_cluster_size = 4`, and 200 per species for the bird
subsamples with `min_cluster_size = 70`.

Collections in which many species fall below that floor produce a high
abstention rate rather than incorrect labels. Abstention is reported as the
noise fraction, and must be read alongside any accuracy figure, because
accuracy is computed over clustered images only.

## 3. Candidate species list, required for naming

Clustering requires no species information. Naming requires a list of candidate
species names, which is the vocabulary the zero-shot matcher may choose from.
It states nothing about which image belongs to which species, so it is not a
label set.

Three sources are supported, selected by `annotation.species_source`:

| Source | Behaviour |
|---|---|
| `dataset` | Candidates are the class directory names |
| `file` | Candidates are read from a CSV |
| `gbif` | Candidates are retrieved from the GBIF occurrence API for a region |

A CSV requires a `class_name` column; `genus`, `family` and `order` are
optional:

```csv
class_name,genus,family,order
Capreolus capreolus,Capreolus,Cervidae,Artiodactyla
Vulpes vulpes,Vulpes,Canidae,Carnivora
```

The prompt is a single structured template, `"a photo of {order} {family}
{genus} {name}"`, and ranks absent from the table are omitted from it. A
regional or national checklist is sufficient. A list broader than the species
actually present is acceptable, since unused candidates attract no clusters.

## 4. Optional metadata

The pipeline runs without metadata. The fields below extend what can be
measured rather than what can be run.

| Field | What it enables |
|---|---|
| `image_id` | Joining a supplied table to the image files |
| `event_id` or `sequence_id` | Grouping frames from one camera-trap trigger, so results can be reported per event as well as per image (Section 5) |
| `timestamp` | Reconstructing events where no event identifier exists; date precision is sufficient |
| `camera_id` or `site_id` | Testing whether clusters correspond to recording sites rather than species; pseudonymous identifiers are sufficient |
| Day or infrared flag | Reporting daylight and infrared frames separately |
| Crop or full frame | Interpreting the effect of detector cropping (Section 5) |
| Verified species for a subset | Measuring accuracy; a few hundred images is sufficient |

Geographic coordinates, site names, personal data and EXIF are not read at any
stage. The pipeline operates on image content alone.

## 5. Camera-trap collections

Camera-trap imagery has three properties that affect how results are produced
and read.

**Trigger sequences.** One trigger produces several near-identical frames.
Treated as independent images, these inflate every clustering metric, because
part of what is recovered is that a frame resembles the frame captured a second
later. An event identifier, or a timestamp from which events can be
reconstructed, allows results to be reported per event.

**Framing.** A full frame in which the animal occupies a small part of the
image is dominated by background, and clusters may then reflect the recording
site rather than the species. Crops produced by an animal detector are expected
to cluster better. This has not been measured in the present work.

**Empty frames.** Collections that retain blank triggers produce a large
cluster of empty frames. Whether blanks have been removed should be recorded,
because it changes the interpretation of both the cluster count and the noise
fraction.

Frames containing more than one species receive a single cluster assignment,
and so count as errors against any single ground-truth label.

## 6. Processing and data handling

All computation is local. Pretrained weights are downloaded once from Hugging
Face on first use, after which the pipeline runs offline. No image, embedding
or result is transmitted to an external service. The optional GBIF lookups
transmit species names only: one resolves a name to its accepted taxonomy, and
one collects reference photographs of a proposed species for the review
report. Both can be disabled, the first by supplying a taxonomy CSV, which
removes network access entirely.

No model is trained or fine-tuned. The image encoder is frozen, so nothing
derived from a collection is retained in any model.

Two properties are relevant where imagery cannot be shared:

- Extraction can be separated from analysis. The extraction stage writes a
  single `.npz` of embeddings, and the clustering, naming and triage stages
  read that file alone. Analysis can therefore proceed without access to the
  images, with the exception of the review report, which renders thumbnails.
- Embeddings are derived data, not anonymised data. They are not viewable as
  images, but approximate reconstruction from neural embeddings is an active
  research area, so an embedding file warrants the same handling as the imagery
  it was computed from.

## 7. Outputs

For each clustering method the pipeline writes a cluster assignment per image,
a proposed species name per cluster, a confidence tier per image
(`auto_accept`, `review`, `human_required`), a review template listing the
images that require attention, diagnostic figures, and optionally a
self-contained HTML review report. Where ground-truth labels are present, it
also reports ARI, NMI, Hungarian accuracy, the recovered cluster count and the
noise fraction.
