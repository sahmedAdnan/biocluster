# The pipeline, stage by stage

Five stages. Each writes to disk, so any stage can be re-run without repeating
the one before it.

```
images ─▶ 1. extract ─▶ 2. reduce ─▶ 3. cluster ─▶ 4. name ─▶ 5. triage ─▶ review
          embeddings     low-dim       cluster ids   species    confidence
          (cached)                                   per        tier per
                                                     cluster    image
```

## 1. Extract embeddings

`biocluster-extract --config configs/paper/<dataset>/<file>.yaml`

A frozen encoder maps each image to one vector. Supported encoders, selected by
`model.name`:

| key | checkpoint | dim | text encoder |
|---|---|---|---|
| `bioclip2.5` | `hf-hub:imageomics/bioclip-2.5-vith14` | 1024 | yes |
| `bioclip2` | `hf-hub:imageomics/bioclip-2` | 768 | yes |
| `bioclip` | `hf-hub:imageomics/bioclip` | 512 | yes |
| `clip` | OpenAI ViT-L/14 | 768 | yes |
| `dinov2` | `vit_large_patch14_dinov2.lvd142m` | 1024 | no |
| `dinov3` | `vit_huge_plus_patch16_dinov3.lvd1689m` | 1280 | no |

Nothing is trained or fine-tuned. Embeddings are L2-normalised and cached on
disk, keyed by the model and by every file's path, size and mtime, so a second
run over the same folder never touches the GPU again. Vision-only encoders
(DINOv2/v3) can cluster but cannot name clusters, because naming needs a text
tower.

## 2. Reduce

`reduction.method`: `none`, `pca`, `umap`, or `pca+umap` (PCA first, then UMAP).
Reduction applies to **clustering only**. Text-image matching in stage 4 always
runs in the original embedding space. Reducing before text matching destroys
the vision-language alignment the encoder was trained for.

## 3. Cluster

Six algorithms behind one interface (`clustering.methods.run_clustering`):
k-means, agglomerative, spectral, GMM, DBSCAN, HDBSCAN. The density methods
(DBSCAN, HDBSCAN) do not need a cluster count and may label points `-1`, meaning
*no cluster*, an abstention, which matters for how results are read (see
below).

## 4. Name the clusters

`biocluster-annotate --config configs/paper/<dataset>/<file>.yaml`

Each cluster centroid is matched against text embeddings of candidate species
names; the nearest name is proposed for the whole cluster. The candidate list is
a vocabulary, not labels: it can be the dataset's own class names, a CSV
checklist, or a GBIF query for a region. Genus, family and order are filled into the prompt when available, which
matches the taxonomic text the BioCLIP family was trained on. Ranks that
are missing are omitted.

## 5. Triage by uncertainty

Four signals per image, combined as a weighted mean (defaults in brackets):

| signal | meaning | weight |
|---|---|---|
| semantic | `1 - (top1 - top2)` cosine margin between candidate names | 0.35 |
| spatial | `1 - HDBSCAN membership probability` (silhouette fallback) | 0.25 |
| cluster agreement | fraction of the cluster voting for a different name | 0.25 |
| taxonomic consistency | whether the cluster mixes families | 0.15 |

Two thresholds split the result into `auto_accept`, `review` and
`human_required`. The thresholds are absolute, not percentile-based: a confident
run may flag nothing, a poor one may flag hundreds. Percentile thresholds would
hide exactly that difference.

## 6. Review

`biocluster-annotate --report` writes a single self-contained `review_<method>.html`
that opens in any browser. Clusters appear worst-first by tier, each with its
proposed name, its runner-up, the four signal values, and reference
photographs for both names. A reviewer can confirm a cluster, accept the
runner-up, relabel individual images, rescue an image out of noise, or type a
name that was not in the candidate list. Finalising downloads one
`file_path,final_label,status` CSV, which `apply_decisions()` turns into a
sorted image folder.

Reference photographs come from one of two places, set by
`annotation.reference_source`:

| source | where the photographs come from | use |
|---|---|---|
| `dataset` | the collection's own class directories | inspecting what the clustering did. Needs ground-truth directories, and is not an independent check of a proposed name, because the proposal was derived from these images |
| `gbif` | GBIF occurrence media for the proposed species | checking a proposed name against imagery from outside the collection. The only option on an unlabelled collection |

Three things about the GBIF option.

**Life stage is filtered, and shown.** GBIF's occurrence media is weighted
towards whichever stage gets photographed, which for Lepidoptera is the
caterpillar: of the first twenty image records for *Noctua pronuba*, eighteen
are larvae. An unfiltered lookup therefore hands a reviewer caterpillar photos
to check adult trap captures against. The lookup requests
`annotation.gbif.life_stages`, default `[Adult, Imago]`, which are the two
Darwin Core values for the insect adult. Immature records are dropped even
when a record carries no stage filter, and every photograph is captioned in
the report with its recorded stage, or with "stage not recorded" where GBIF
holds none. Set `life_stages` differently for taxa where another stage is what
gets imaged.

**Cost follows the cluster count.** Only the names the report displays are
looked up, which is each cluster's top choice and its runner-up, not the whole
candidate list. Results are cached, so a repeated run makes no requests.

**A photograph is not a determination.** GBIF occurrence media is contributed
by observers and institutions, and an occurrence's identification is not
necessarily expert-verified, so these support a visual comparison rather than
a determination. The name itself is checked separately against the GBIF
backbone taxonomy, which returns a match type and a confidence score; matches
below 70 are reported as such.

Only species names are sent to GBIF. No image, embedding or result leaves the
machine.

## Reading the output honestly

Two properties of this design will mislead you if ignored.

**Noise points receive a tier.** An image HDBSCAN left unclustered gets maximal
spatial, agreement and taxonomic uncertainty, so it lands in `human_required`
with the literal name `Noise / Unknown`. The tier counts therefore include
abstentions.

**Accuracy excludes noise.** Per-image accuracy is computed over clustered
points only. So a configuration that abstains heavily looks more accurate than
one that commits: it is being graded on the easy subset. Always read accuracy
next to the retained fraction. For the same reason, selecting a configuration on
accuracy alone will favour whichever setting discards the most data.
