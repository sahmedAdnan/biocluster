import sys
import os
import json
import glob
# Deterministic cuBLAS: must be set BEFORE torch initializes CUDA, so results
# (feature extraction included) are reproducible across runs. See set_seed().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import traceback as _traceback
from functools import partial
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QComboBox,
                               QFileDialog, QGroupBox, QFormLayout, QProgressBar,
                               QFrame, QMessageBox, QDoubleSpinBox, QSpinBox,
                               QTabWidget, QScrollArea, QGridLayout, QCheckBox,
                               QSplashScreen, QTableWidget, QTableWidgetItem, QDialog,
                               QHeaderView, QSplitter, QProgressDialog)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor
from PySide6.QtWebEngineWidgets import QWebEngineView
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import qdarktheme

# The logo and launch emblem ship as package data beside this file. PyInstaller
# unpacks them into _MEIPASS instead, so the frozen bundle needs its own branch.
if getattr(sys, 'frozen', False):
    _LOGO_DIR = os.path.join(sys._MEIPASS, 'assets')
else:
    _LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

from biocluster.app import theme as T
from biocluster.app import feature_cache
from biocluster.app.components import FlowBar


def _set_role(btn, role):
    """Style a button through the theme's role property rather than an inline
    stylesheet. Inline hexes bypassed the palette (so they never followed the
    dark theme) and each carried its own padding, which is why one toolbar
    held three different button heights."""
    btn.setProperty("role", role)
    btn.style().unpolish(btn)
    btn.style().polish(btn)

from biocluster.models.extractors import build_extractor
from biocluster.reduction.reduction import reduce_features
from biocluster.clustering.methods import run_clustering
from biocluster.data.dataset import load_imagefolder
from biocluster.evaluation.metrics import hungarian_accuracy
from biocluster.annotation.annotator import run_annotation
from biocluster.annotation.taxonomy import load_taxonomy_csv, parse_taxonomy_from_names
from biocluster.utils.io import set_seed

# ---------------------------------------------------------------------------
# Persistent run history: every completed run is appended here so settings and
# results are never lost when the app closes. Lives in the user's home so it
# survives app rebuilds/reinstalls.
# ---------------------------------------------------------------------------
_HISTORY_DIR = os.path.join(os.path.expanduser("~"), "BioCluster")
_HISTORY_CSV = os.path.join(_HISTORY_DIR, "run_history.csv")
_HISTORY_COLUMNS = [
    "timestamp", "dataset", "n_images", "n_clusters", "noise_pct",
    "model", "device", "labeled", "batch_size", "num_workers",
    "reduction", "reduction_params", "clustering", "clustering_params", "seed",
    "ARI", "NMI", "Hungarian", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin",
    "folder",
]


# ---------------------------------------------------------------------------
# Model registry: one entry per item in the Model dropdown.
#   pretrained : identifier handed to build_extractor
#   arch       : open_clip architecture name. Required when `pretrained` is a
#                bare weight tag ('openai') rather than an 'hf-hub:' URI. The
#                BioCLIP checkpoints carry their architecture in the URI, plain
#                CLIP does not and fails to load without this.
#   has_text   : model has a text encoder, so the annotation step can run.
#                DINOv2 and DINOv3 are vision-only.
# ---------------------------------------------------------------------------
_MODEL_SPECS = {
    "bioclip2.5": {"pretrained": "hf-hub:imageomics/bioclip-2.5-vith14",  "has_text": True},
    "bioclip2":   {"pretrained": "hf-hub:imageomics/bioclip-2",           "has_text": True},
    "bioclip":    {"pretrained": "hf-hub:imageomics/bioclip",             "has_text": True},
    "clip":       {"pretrained": "openai", "arch": "ViT-L-14",            "has_text": True},
    "dinov2":     {"pretrained": "vit_large_patch14_dinov2.lvd142m",      "has_text": False},
    "dinov3":     {"pretrained": "vit_huge_plus_patch16_dinov3.lvd1689m", "has_text": False},
}


def model_has_text(key: str) -> bool:
    """True when the model can encode text, i.e. annotation is available."""
    return bool(_MODEL_SPECS.get(key, {}).get("has_text", False))


def build_model_config(key: str, batch_size: int, num_workers: int,
                       device: str, global_pool: str = None,
                       img_size: int = None) -> dict:
    """Assemble the config['model'] block for a dropdown key.

    Single source of truth: the pipeline and annotation paths both call this,
    so a model added here cannot work in one and fail in the other."""
    spec = _MODEL_SPECS.get(key) or _MODEL_SPECS["bioclip"]
    block = {
        "name": key,
        "pretrained": spec["pretrained"],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "normalize_embeddings": True,
    }
    if spec.get("arch"):
        block["arch"] = spec["arch"]
    # timm-only knobs; absent means "use the checkpoint's own default".
    # Guarded on the model family: the BioCLIP/CLIP path goes through open_clip,
    # which accepts neither, so passing them there would be silently misleading.
    if model_is_timm(key):
        if global_pool:
            block["global_pool"] = global_pool
        if img_size:
            block["img_size"] = int(img_size)
    return {"model": block}


def model_is_timm(key: str) -> bool:
    """True for the DINO family, loaded through timm and therefore able to
    accept the global_pool / img_size overrides."""
    return key in ("dinov2", "dinov3")


def extraction_opts(key: str, global_pool: str = None, img_size: int = None) -> dict:
    """Model settings that change the EMBEDDINGS, not just the speed. Fed into
    the cache fingerprint so a pooling change can never be served a stale
    entry."""
    if not model_is_timm(key):
        return {}
    return {"global_pool": global_pool or "", "img_size": img_size or ""}


def _run_identity(row: dict):
    """Identity of a run for dedup: full config + seed. Params are normalised
    (sorted keys) so ordering never causes a false mismatch. Two runs with the
    same identity are deterministic replicas → keep only the latest."""
    import json

    def norm(s):
        try:
            return json.dumps(json.loads(s or "{}"), sort_keys=True)
        except Exception:
            return str(s or "")

    fields = ("dataset", "model", "device", "labeled",
              "reduction", "reduction_params", "clustering", "clustering_params")
    return tuple(
        norm(row.get(f, "")) if f.endswith("_params") else str(row.get(f, ""))
        for f in fields
    ) + (str(row.get("seed", "")),)


def write_run_history(row: dict, path: str = _HISTORY_CSV) -> str:
    """Record one run's config+metrics. If a run with the same identity
    (config + seed) already exists it is UPDATED in place (dedup); otherwise a
    new row is appended. Writes atomically (temp file + replace) so a failure
    can never corrupt the log, and falls back to a plain append if the existing
    file can't be read (never truncates data it couldn't load). Module-level so
    it is unit-testable without a running GUI."""
    import csv
    import tempfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    newrow = {c: row.get(c, "") for c in _HISTORY_COLUMNS}

    # Try to read the existing log so we can dedup. A read failure must NOT
    # cause data loss, so in that case we fall back to a safe append.
    existing = []
    read_ok = True
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        except Exception:
            read_ok = False

    if not read_ok:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_HISTORY_COLUMNS, extrasaction="ignore")
            writer.writerow(newrow)
        return path

    ident = _run_identity(newrow)
    replaced = False
    for i, r in enumerate(existing):
        if _run_identity(r) == ident:
            existing[i] = newrow           # deterministic replica -> update
            replaced = True
            break
    if not replaced:
        existing.append(newrow)

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_HISTORY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in existing:
                writer.writerow({c: r.get(c, "") for c in _HISTORY_COLUMNS})
        os.replace(tmp, path)              # atomic swap
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise
    return path

class _SplashScreen(QSplashScreen):
    """Splash screen rendered from Launching_emblem.png with a status line."""

    def __init__(self, pixmap: QPixmap):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self._msg = ""

    def showMessage(self, message, alignment=Qt.AlignBottom, color=Qt.black):
        self._msg = message
        self.repaint()

    def drawContents(self, painter):
        if not self._msg:
            return
        painter.save()
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        painter.setPen(QColor("#2f3542"))
        rect = self.rect().adjusted(20, 0, -20, -18)
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignHCenter, self._msg)
        painter.restore()


class _SpinBox(QSpinBox):
    def wheelEvent(self, event): event.ignore()

class _DoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event): event.ignore()

class _ComboBox(QComboBox):
    def wheelEvent(self, event): event.ignore()


class PipelineWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str, str)   # (message, traceback)
    cancelled = Signal()

    def __init__(self, folder, has_labels, extractor_name, reduction_method, reduction_params,
                 clustering_method, clustering_params, device, batch_size, num_workers,
                 shared_extractor=None, model_key=None,
                 global_pool=None, img_size=None):
        super().__init__()
        self.folder = folder
        self.has_labels = has_labels
        self.extractor_name = extractor_name
        self.reduction_method = reduction_method
        self.reduction_params = reduction_params
        self.clustering_method = clustering_method
        self.clustering_params = clustering_params
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_fp = None
        self.from_disk_cache = False
        self.global_pool = global_pool
        self.img_size = img_size
        # Model reuse: if the window already has this exact model loaded,
        # it is passed in here so we skip the (slow) reload. After a build,
        # `self.extractor` exposes the model so the window can cache it.
        self.shared_extractor = shared_extractor
        self.model_key = model_key
        self.extractor = None

    def run(self):
        try:
            extractor = None   # kept alive and reused by the annotation step

            # Reproducibility: seed Python/NumPy/PyTorch(+CUDA) and force
            # deterministic kernels BEFORE any extraction, so the same settings
            # + seed reproduce the same score (feature extraction included).
            seed = self.reduction_params.get("random_state",
                                             self.clustering_params.get("random_state", 42))
            set_seed(seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

            # ALWAYS rescan the folder. There used to be an in-memory cache
            # keyed on (folder, model, has_labels) that skipped this entirely,
            # so adding or deleting images in an already-run folder was
            # invisible until the app restarted -- it silently reported results
            # for the previous file list. Scanning is a directory listing and
            # costs milliseconds; the on-disk cache below fingerprints the file
            # list (path + size + mtime), so a genuinely unchanged folder still
            # skips extraction while any change correctly invalidates it.
            self.progress.emit(10, "Scanning folder...")

            true_labels = None
            class_names = None

            if self.has_labels:
                try:
                    extensions = ["jpg", "jpeg", "png", "webp"]
                    image_paths, true_labels, class_names = load_imagefolder(self.folder, extensions)
                    true_labels = np.array(true_labels)
                except Exception as e:
                    raise ValueError(f"Could not load labels. Ensure your folder has subfolders representing classes. Error: {str(e)}")
            else:
                extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
                image_paths = []
                for ext in extensions:
                    image_paths.extend(glob.glob(os.path.join(self.folder, '**', ext), recursive=True))
                image_paths = sorted(image_paths)

            if not image_paths:
                raise ValueError("No images found in the selected folder.")

            if self.isInterruptionRequested():
                self.cancelled.emit(); return

            # Disk cache: extraction depends only on (model, file list), so
            # a folder already extracted with this model never needs the GPU
            # again -- across restarts, and regardless of batch size.
            self.cache_fp = feature_cache.fingerprint(
                self.extractor_name, image_paths,
                opts=extraction_opts(self.extractor_name,
                                     self.global_pool, self.img_size))
            hit = feature_cache.load(self.cache_fp)
            if hit is not None:
                features, cached_labels, _, cached_classes = hit
                if self.has_labels and true_labels is None and cached_labels:
                    true_labels = np.array(cached_labels)
                if class_names is None and cached_classes:
                    class_names = list(cached_classes)
                self.progress.emit(
                    48, f"Loaded cached features ({features.shape[0]} images) "
                        f"— skipping extraction")
                self.from_disk_cache = True

            else:
                # Cache miss -- load the model and extract for real.
                # Reuse an already-loaded model when the window passes one
                # in (same model + device); otherwise build it once here.
                if self.shared_extractor is not None:
                    self.progress.emit(20, f"Reusing loaded {self.extractor_name} model...")
                    extractor = self.shared_extractor
                    # The extractor reads its run settings from itself, and they
                    # were fixed when it was built. Without this a reused model
                    # silently kept the first run's batch size, while the cache
                    # index and run history recorded the new one.
                    extractor.batch_size = self.batch_size
                    extractor.num_workers = self.num_workers
                else:
                    self.progress.emit(20, f"Loading {self.extractor_name} model...")
                    config = build_model_config(
                        self.extractor_name, self.batch_size,
                        self.num_workers, self.device,
                        global_pool=self.global_pool, img_size=self.img_size)
                    extractor = build_extractor(config)
                # Expose the (built or reused) model so the window can cache it.
                self.extractor = extractor

                self.progress.emit(22, f"Extracting features using {self.extractor_name}...")

                def _extract_progress(done, total):
                    pct = 22 + int(26 * done / total)
                    self.progress.emit(pct, f"Extracting features... batch {done}/{total}")

                features = extractor.extract(image_paths,
                                             progress_fn=_extract_progress,
                                             cancelled_fn=self.isInterruptionRequested)
                if features is None:
                    self.cancelled.emit(); return

                feature_cache.save(
                    self.cache_fp, features, true_labels, image_paths, class_names,
                    meta={"model": self.extractor_name,
                          "folder": self.folder,
                          "global_pool": self.global_pool or "default",
                          "img_size": self.img_size or "native",
                          # Not part of the cache key, but recorded so an entry
                          # can be traced back to how it was produced.
                          "batch_size": self.batch_size,
                          "num_workers": self.num_workers,
                          "device": self.device,
                          "cublas_workspace_config":
                              os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                          "n_images": int(features.shape[0])})

                # Release transient GPU memory (activations/buffers) after
                # extraction. The model weights stay resident for reuse; this
                # just prevents the allocator from holding fragmented blocks
                # that would otherwise spill to slow shared memory next run.
                try:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if self.isInterruptionRequested():
                self.cancelled.emit(); return

            self.progress.emit(50, f"Reducing features using {self.reduction_method}...")
            
            red_config = {
                "reduction": {
                    "method": self.reduction_method,
                    **self.reduction_params
                }
            }
            seed = self.reduction_params.get("random_state", 42)
            reduced_features, _ = reduce_features(features, red_config, random_state=seed)

            if self.isInterruptionRequested():
                self.cancelled.emit(); return

            self.progress.emit(70, f"Clustering using {self.clustering_method}...")

            # Capture HDBSCAN membership probabilities so the annotation step can
            # use them as the spatial uncertainty signal (parity with the full
            # pipeline). Other methods have no membership probability -> None,
            # and annotation falls back to silhouette.
            probabilities = None
            if self.clustering_method == "hdbscan":
                cluster_ids, probabilities = run_clustering(
                    self.clustering_method, reduced_features,
                    self.clustering_params, return_probabilities=True)
            else:
                cluster_ids = run_clustering(
                    self.clustering_method, reduced_features, self.clustering_params)

            self.progress.emit(85, "Calculating Evaluation Metrics...")
            
            metrics = {}
            valid_mask = cluster_ids != -1
            
            if self.has_labels:
                metrics['type'] = 'external'
                if valid_mask.sum() > 0:
                    t = true_labels[valid_mask]
                    c = cluster_ids[valid_mask]
                    metrics['ARI'] = adjusted_rand_score(t, c)
                    metrics['NMI'] = normalized_mutual_info_score(t, c)
                    acc, _ = hungarian_accuracy(t, c)
                    metrics['Hungarian Accuracy'] = acc
                else:
                    metrics['ARI'] = "N/A"
                    metrics['NMI'] = "N/A"
                    metrics['Hungarian Accuracy'] = "N/A"
            else:
                metrics['type'] = 'internal'
                if valid_mask.sum() > 1 and len(np.unique(cluster_ids[valid_mask])) > 1:
                    metrics['Silhouette Score'] = silhouette_score(reduced_features[valid_mask], cluster_ids[valid_mask])
                    metrics['Calinski-Harabasz'] = calinski_harabasz_score(reduced_features[valid_mask], cluster_ids[valid_mask])
                    metrics['Davies-Bouldin'] = davies_bouldin_score(reduced_features[valid_mask], cluster_ids[valid_mask])
                else:
                    metrics['Silhouette Score'] = "N/A (Needs >1 cluster)"
                    metrics['Calinski-Harabasz'] = "N/A (Needs >1 cluster)"
                    metrics['Davies-Bouldin'] = "N/A (Needs >1 cluster)"

            self.progress.emit(90, "Generating 2D Projection for Visualization...")
            
            # Always generate a strict 2D projection for the visualization
            if reduced_features.shape[1] > 2:
                viz_config = {
                    "reduction": {
                        "method": "umap",
                        "n_components": 2,
                        "n_neighbors": 15,
                        "min_dist": 0.1,
                        "metric": "cosine"
                    }
                }
                viz_features, _ = reduce_features(reduced_features, viz_config, random_state=seed)
            elif reduced_features.shape[1] == 1:
                viz_features = np.column_stack((reduced_features[:, 0], np.zeros(len(reduced_features))))
            else:
                viz_features = reduced_features

            self.progress.emit(100, "Pipeline complete!")
            
            self.finished.emit({
                "image_paths": image_paths,
                "features":    features,
                "viz_features": viz_features,
                "cluster_ids": cluster_ids,
                "probabilities": probabilities,
                "metrics": metrics,
                "extractor": extractor,               # None if features were cached
                "extractor_model": self.extractor_name,
                "class_names": class_names if self.has_labels else None,
                "true_labels": true_labels if self.has_labels else None
            })

        except Exception as e:
            self.error.emit(str(e), _traceback.format_exc())


class AnnotationWorker(QThread):
    """Runs zero-shot annotation on an already-clustered result.

    Reuses biocluster.annotation: encode candidate species text, match cluster
    centroids, compute the four uncertainty signals and confidence tiers.
    Emits the annotations DataFrame (one row per image).
    """
    progress = Signal(int, str)
    finished = Signal(object)      # pandas DataFrame
    error = Signal(str, str)

    def __init__(self, features, cluster_ids, image_paths, candidate_source,
                 extractor_name, device, batch_size,
                 weights, review_threshold, human_threshold,
                 taxonomy_csv=None, folder_class_names=None,
                 probabilities=None, extractor=None):
        super().__init__()
        self.features = features
        self.cluster_ids = cluster_ids
        self.image_paths = image_paths
        self.probabilities = probabilities
        self.extractor = extractor          # reuse a loaded model if provided
        self.candidate_source = candidate_source
        self.extractor_name = extractor_name
        self.device = device
        self.batch_size = batch_size
        self.weights = weights                       # configurable S/P/A/T weights
        self.review_threshold = review_threshold
        self.human_threshold = human_threshold
        self.taxonomy_csv = taxonomy_csv
        self.folder_class_names = folder_class_names

    def run(self):
        try:
            # 1. Candidate species list + taxonomy table
            self.progress.emit(10, "Building candidate list...")
            if self.candidate_source == "csv":
                taxonomy_table = load_taxonomy_csv(self.taxonomy_csv)
                species = list(taxonomy_table.keys())
            else:  # folder labels
                species = list(self.folder_class_names or [])
                taxonomy_table = parse_taxonomy_from_names(species)
            if not species:
                raise ValueError("No candidate species found for annotation.")

            # 2. Text encoder (BioCLIP). Reuse the model already loaded by the
            #    pipeline run if available; otherwise load it (HF-cached on disk).
            if self.extractor is not None:
                self.progress.emit(45, "Reusing loaded model...")
                extractor = self.extractor
            else:
                self.progress.emit(30, "Loading text encoder...")
                config = build_model_config(
                    self.extractor_name, self.batch_size, 0, self.device)
                extractor = build_extractor(config)

            # 3. Prompt: single taxonomic prompt when family taxonomy exists
            #    (equal-best in ablation, in-distribution for BioCLIP-2), else
            #    a generic single prompt. No ensemble (it lowered accuracy).
            has_family = any(v.get("family") for v in taxonomy_table.values())
            templates = (["a photo of {order} {family} {genus} {name}"]
                         if has_family else ["a photo of {name}"])
            self.progress.emit(55, f"Encoding {len(species)} species...")
            text_embs = extractor.encode_text(
                species, prompt_templates=templates,
                taxonomy_table=taxonomy_table)

            # 4. Annotate clusters + 4 uncertainty signals + tiers.
            #    IMPORTANT: original (un-reduced) features: text matching only
            #    works in the full CLIP embedding space.
            self.progress.emit(80, "Matching clusters + uncertainty signals...")
            ann_config = {"annotation": {"uncertainty": {
                "review_threshold": self.review_threshold,
                "human_threshold": self.human_threshold,
                "weights": self.weights,
            }}}
            df = run_annotation(
                self.features, self.cluster_ids, text_embs, species,
                taxonomy_table, ann_config,
                probabilities=self.probabilities, image_paths=self.image_paths)

            self.progress.emit(100, "Annotation complete")
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e), _traceback.format_exc())


class _ClickableLabel(QLabel):
    """QLabel that emits clicked(path) on press -- used for zoomable thumbnails."""
    clicked = Signal(str)

    def __init__(self, path="", parent=None):
        super().__init__(parent)
        self._path = path
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if self._path:
            self.clicked.emit(self._path)


class _CreditsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        import html
        from biocluster import __version__
        from biocluster.app import credits as C

        self.setWindowTitle("About Bio Cluster")
        self.resize(600, 860)
        esc = html.escape
        muted = f"color:{T.INK_MUTED};"

        def link(url, text):
            return (f'<a href="{esc(url)}" style="color:{T.ACCENT}; '
                    f'text-decoration:none;">{esc(text)}</a>')

        def rich(text, name=None):
            lbl = QLabel(text)
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setOpenExternalLinks(True)
            lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
            if name:
                lbl.setObjectName(name)
            return lbl

        def card(eyebrow, *widgets):
            frame = QFrame()
            frame.setObjectName("Card")
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(T.SPACE_LG, T.SPACE_MD, T.SPACE_LG, T.SPACE_LG)
            lay.setSpacing(T.SPACE_MD)
            cap = QLabel(eyebrow.upper())
            cap.setObjectName("Eyebrow")
            lay.addWidget(cap)
            for w in widgets:
                lay.addWidget(w)
            return frame

        body = QWidget()
        body.setObjectName("CreditsBody")
        body.setStyleSheet(f"QWidget#CreditsBody {{ background: {T.WINDOW}; }}")
        col = QVBoxLayout(body)
        col.setContentsMargins(T.SPACE_XL, T.SPACE_XL, T.SPACE_XL, T.SPACE_LG)
        col.setSpacing(T.SPACE_MD)

        # Header: logo, name, version and what it is.
        head = QHBoxLayout()
        head.setSpacing(T.SPACE_LG)
        pix = QPixmap(os.path.join(_LOGO_DIR, "app_logo.png"))
        if not pix.isNull():
            logo = QLabel()
            logo.setPixmap(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            head.addWidget(logo, 0, Qt.AlignTop)
        name_col = QVBoxLayout()
        name_col.setSpacing(2)
        title = QLabel("Bio Cluster")
        title.setObjectName("PageTitle")
        version = QLabel(f"Version {__version__}")
        version.setObjectName("Caption")
        tagline = QLabel(C.TAGLINE)
        tagline.setObjectName("Muted")
        tagline.setWordWrap(True)
        for w in (title, version, tagline):
            name_col.addWidget(w)
        head.addLayout(name_col, 1)
        col.addLayout(head)
        col.addSpacing(T.SPACE_SM)

        devs = QLabel(" and ".join(C.DEVELOPERS))
        devs.setObjectName("CardTitle")
        col.addWidget(card("Developed by", devs))

        authors = []
        for a in C.AUTHORS:
            orcid = ("<br>" + link("https://orcid.org/" + a["orcid"], "orcid.org/" + a["orcid"])
                     if a["orcid"] else "")
            authors.append(rich(f'<b>{esc(a["name"])}</b><br>'
                                f'<span style="{muted}">{esc(a["affiliation"])}</span>{orcid}'))
        col.addWidget(card("Authors", *authors))

        built = []
        for name, by, url in C.BUILT_ON:
            item = link(url, name) if url else esc(name)
            if by:
                item += f' <span style="{muted}">· {esc(by)}</span>'
            built.append(item)
        col.addWidget(card("Built on", rich("<br>".join(built))))
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)

        footer = QWidget()
        footer.setObjectName("CreditsFooter")
        footer.setStyleSheet(f"QWidget#CreditsFooter {{ background: {T.SURFACE}; "
                             f"border-top: 1px solid {T.BORDER}; }}")
        foot = QHBoxLayout(footer)
        foot.setContentsMargins(T.SPACE_XL, T.SPACE_MD, T.SPACE_XL, T.SPACE_MD)
        foot.addWidget(rich(f'<span style="{muted}">{esc(C.LICENSE)}</span>'
                            f' &nbsp;·&nbsp; {link(C.REPOSITORY, "Source on GitHub")}', "Caption"), 1)
        close = QPushButton("Close")
        close.setMinimumWidth(96)
        close.setDefault(True)
        close.clicked.connect(self.accept)
        foot.addWidget(close)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        outer.addWidget(footer)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio Cluster")
        self.setMinimumSize(1400, 900)
        _logo = os.path.join(_LOGO_DIR, "app_logo.png")
        if os.path.exists(_logo):
            self.setWindowIcon(QIcon(_logo))
        self.reduction_widgets = {}
        self.clustering_widgets = {}
        # Keep a single loaded model resident across runs, keyed by
        # (model_name, device). Reused when unchanged (fast); freed and
        # replaced when the user switches models (avoids VRAM spillover).
        self._cached_extractor = None
        self._cached_extractor_key = None
        self._gallery_shown = 60
        self.folder_path = None
        self.annotation_df = None
        self._ann_lookup = {}
        self.taxonomy_csv_path = None
        self.decisions = {}                    # file_path -> {"final","status"} corrections
        self._candidates = []                  # species options for relabel combos
        self._tier_filter = "all"              # gallery cluster filter
        self._cluster_tier = {}                # cluster id -> worst per-image tier
        self._present_clusters = []            # all cluster names shown in the dropdown
        self._undo = []                        # decision-history snapshots (JSON)
        self._redo = []
        self._autosave_path = None             # per-dataset autosave file
        self._ref_map = {}                     # species -> dataset image paths (references)
        self._img_correct = {}                 # file_path -> bool (annotation vs ground truth)
        self._overall_acc = None               # overall per-image annotation accuracy
        self._cluster_acc = None               # overall cluster-naming accuracy
        self._overall_acc_n = (0, 0)           # (correct, total) images
        self._cluster_acc_n = (0, 0)           # (correct, total) clusters
        self._cached_extractor = None          # reused by annotation to skip reload
        self._cached_extractor_model = None
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left Panel: Configuration Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(15)

        # App Title
        title_lbl = QLabel("Bio Cluster")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title_lbl.setFont(font)
        about_lbl = QLabel(f'<a href="#credits" style="color:{T.ACCENT}; '
                           f'text-decoration:none;">About &amp; credits</a>')
        about_lbl.setObjectName("Caption")
        about_lbl.linkActivated.connect(self.show_credits)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(title_lbl)
        title_box.addWidget(about_lbl)
        sidebar_layout.addLayout(title_box)

        data_group = QGroupBox("1. Data Input")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)
        
        self.btn_browse = QPushButton("Browse Image Folder")
        self.btn_browse.setMinimumHeight(35)
        self.btn_browse.clicked.connect(self.browse_folder)
        
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setWordWrap(True)
        
        self.chk_labels = QCheckBox("Folder has Ground Truth Labels (Subfolders)")
        self.chk_labels.setChecked(False)
        self.chk_labels.setToolTip("Check this if your images are organized into subfolders representing their true classes.")
        
        data_layout.addWidget(self.btn_browse)
        data_layout.addWidget(self.lbl_folder)
        data_layout.addWidget(self.chk_labels)
        data_group.setLayout(data_layout)

        extract_group = QGroupBox("2. Feature Extraction")
        extract_layout = QFormLayout()
        extract_layout.setSpacing(10)
        self.combo_model = _ComboBox()
        self.combo_model.setMinimumHeight(30)
        self.combo_model.addItem("BioCLIP 2.5 Huge (ViT-H/14)", "bioclip2.5")
        self.combo_model.addItem("BioCLIP2", "bioclip2")
        self.combo_model.addItem("BioCLIP (Original)", "bioclip")
        self.combo_model.addItem("CLIP (ViT-L/14)", "clip")
        self.combo_model.addItem("DINOv2 (ViT-L/14)", "dinov2")
        self.combo_model.addItem("DINOv3 (ViT-H+/16)", "dinov3")
        extract_layout.addRow("Model:", self.combo_model)

        # How patch tokens collapse into one embedding. Only the DINO models go
        # through timm, so this is hidden for the CLIP family.
        self.combo_pool = _ComboBox()
        self.combo_pool.setMinimumHeight(30)
        self.combo_pool.addItem("Model default", "")
        self.combo_pool.addItem("CLS token (subject small in frame)", "token")
        self.combo_pool.addItem("Mean patches (subject fills frame)", "avg")
        self.combo_pool.setToolTip("\n".join([
            "How the per-patch features become one vector.",
            "",
            "CLS token: a learned summary that can attend selectively and "
            "ignore background. Better for ordinary photos where the subject "
            "is small in the frame (measured +0.097 ARI on full-frame CUB).",
            "",
            "Mean patches: every patch counts equally, background included. "
            "Better when images are already cropped to the subject "
            "(measured +0.041 ARI on bbox-cropped CUB).",
            "",
            "These do not stack: cropping removes the background that CLS "
            "pooling exists to suppress. Pick one based on your images.",
        ]))
        self.lbl_pool = QLabel("Pooling:")
        extract_layout.addRow(self.lbl_pool, self.combo_pool)
        self.combo_model.currentIndexChanged.connect(self._update_pool_visibility)
        extract_group.setLayout(extract_layout)

        reduce_group = QGroupBox("3. Dimensionality Reduction")
        reduce_layout = QVBoxLayout()
        reduce_layout.setSpacing(10)
        self.combo_reduction = _ComboBox()
        self.combo_reduction.setMinimumHeight(30)
        self.combo_reduction.addItems(["pca", "umap", "pca+umap", "none"])
        self.combo_reduction.currentTextChanged.connect(self._update_reduction_params)
        self.combo_reduction.currentTextChanged.connect(self._update_seed_scope)
        
        reduce_form = QFormLayout()
        reduce_form.setSpacing(10)
        reduce_form.addRow("Method:", self.combo_reduction)
        
        self.reduction_params_layout = QFormLayout()
        self.reduction_params_layout.setSpacing(10)
        
        reduce_layout.addLayout(reduce_form)
        reduce_layout.addLayout(self.reduction_params_layout)
        reduce_group.setLayout(reduce_layout)

        cluster_group = QGroupBox("4. Clustering")
        cluster_layout = QVBoxLayout()
        cluster_layout.setSpacing(10)
        self.combo_clustering = _ComboBox()
        self.combo_clustering.setMinimumHeight(30)
        self.combo_clustering.addItems(["kmeans", "dbscan", "hdbscan", "spectral", "agglomerative", "gmm"])
        self.combo_clustering.currentTextChanged.connect(self._update_cluster_params)
        self.combo_clustering.currentTextChanged.connect(self._update_seed_scope)
        
        cluster_form = QFormLayout()
        cluster_form.setSpacing(10)
        cluster_form.addRow("Method:", self.combo_clustering)
        
        self.clustering_params_layout = QFormLayout()
        self.clustering_params_layout.setSpacing(10)
        
        cluster_layout.addLayout(cluster_form)
        cluster_layout.addLayout(self.clustering_params_layout)
        cluster_group.setLayout(cluster_layout)

        # 7. Annotation (optional, zero-shot species naming). Runs AFTER
        # clustering, so it lives at the bottom, below the Run controls.
        annot_group = QGroupBox("7. Annotation (after clustering)")
        annot_layout = QVBoxLayout()
        annot_layout.setSpacing(10)

        self.combo_candidate = _ComboBox()
        self.combo_candidate.setMinimumHeight(30)
        self.combo_candidate.addItem("Folder labels (ground truth)", "folder")
        self.combo_candidate.addItem("Taxonomy CSV file", "csv")
        self.combo_candidate.currentTextChanged.connect(self._update_candidate_source)
        annot_layout.addWidget(QLabel("Candidate species from:"))
        annot_layout.addWidget(self.combo_candidate)

        self.btn_browse_csv = QPushButton("Browse taxonomy CSV...")
        self.btn_browse_csv.setMinimumHeight(30)
        self.btn_browse_csv.clicked.connect(self.browse_taxonomy_csv)
        self.btn_browse_csv.setVisible(False)
        self.lbl_csv = QLabel("")
        self.lbl_csv.setWordWrap(True)
        self.lbl_csv.setVisible(False)
        self.lbl_csv.setStyleSheet(T.caption_qss())
        annot_layout.addWidget(self.btn_browse_csv)
        annot_layout.addWidget(self.lbl_csv)

        # Off by default because it queries an external API.
        self.chk_gbif_refs = QCheckBox("Web report: GBIF reference photos")
        self.chk_gbif_refs.setToolTip(
            "Show photographs of each proposed species from GBIF beside its "
            "cluster, so a name can be checked against imagery from outside "
            "this collection. Adults only, and each photograph is captioned "
            "with its recorded life stage.\n\n"
            "Sends species names only, never images. Stores URLs, so the "
            "report needs a connection when it is opened.\n\n"
            "GBIF occurrence photographs are contributor-submitted and are "
            "not necessarily expert-verified: they support a visual "
            "comparison, not a determination.")
        annot_layout.addWidget(self.chk_gbif_refs)

        # Configurable uncertainty weights + tier thresholds. Weights need not
        # sum to 1; run_annotation normalizes by their total.
        unc_lbl = QLabel("Uncertainty signal weights:")
        unc_lbl.setStyleSheet(T.caption_qss() + " margin-top: 4px;")
        annot_layout.addWidget(unc_lbl)
        unc_form = QFormLayout()
        unc_form.setSpacing(6)
        self.spin_w_sem = self._create_double_spinbox(0.35, 0.0, 1.0, 0.05)
        self.spin_w_spa = self._create_double_spinbox(0.25, 0.0, 1.0, 0.05)
        self.spin_w_agr = self._create_double_spinbox(0.25, 0.0, 1.0, 0.05)
        self.spin_w_tax = self._create_double_spinbox(0.15, 0.0, 1.0, 0.05)
        self.spin_w_sem.setToolTip("Semantic (text margin) weight")
        self.spin_w_spa.setToolTip("Spatial (membership / silhouette) weight")
        self.spin_w_agr.setToolTip("Cluster-agreement weight")
        self.spin_w_tax.setToolTip("Taxonomic-consistency weight")
        unc_form.addRow("S  semantic:", self.spin_w_sem)
        unc_form.addRow("P  spatial:", self.spin_w_spa)
        unc_form.addRow("A  agreement:", self.spin_w_agr)
        unc_form.addRow("T  taxonomic:", self.spin_w_tax)
        self.spin_rev = self._create_double_spinbox(0.55, 0.0, 1.0, 0.05)
        self.spin_hum = self._create_double_spinbox(0.70, 0.0, 1.0, 0.05)
        self.spin_rev.setToolTip("combined >= this -> 'review'")
        self.spin_hum.setToolTip("combined >= this -> 'human_required'")
        unc_form.addRow("Review threshold:", self.spin_rev)
        unc_form.addRow("Human threshold:", self.spin_hum)
        annot_layout.addLayout(unc_form)

        self.btn_annotate = QPushButton("Annotate Clusters")
        _set_role(self.btn_annotate, "special")
        self.btn_annotate.clicked.connect(self.run_annotation_step)
        self.btn_annotate.setEnabled(False)
        self.btn_annotate.setToolTip("Run the pipeline first. Requires a BioCLIP model (DINOv3 has no text encoder).")
        annot_layout.addWidget(self.btn_annotate)

        self.lbl_annot_status = QLabel("")
        self.lbl_annot_status.setWordWrap(True)
        self.lbl_annot_status.setStyleSheet(T.caption_qss())
        annot_layout.addWidget(self.lbl_annot_status)
        annot_group.setLayout(annot_layout)

        hw_group = QGroupBox("5. Hardware Configuration")
        hw_layout = QFormLayout()
        hw_layout.setSpacing(10)
        
        self.combo_device = _ComboBox()
        self.combo_device.setMinimumHeight(30)
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            self.combo_device.addItem(f"GPU ({gpu_name})", "cuda")
            self.combo_device.addItem("CPU", "cpu")
            default_batch = 64
        else:
            self.combo_device.addItem("CPU (No GPU Detected)", "cpu")
            default_batch = 16
            
        self.spin_batch = _SpinBox()
        self.spin_batch.setMinimumHeight(30)
        self.spin_batch.setRange(1, 1024)
        self.spin_batch.setValue(default_batch)
        
        self.spin_workers = _SpinBox()
        self.spin_workers.setMinimumHeight(30)
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setValue(0)
        self.spin_workers.setToolTip("Set to 0 for maximum stability on Windows.")
        
        hw_layout.addRow("Device:", self.combo_device)
        hw_layout.addRow("Batch Size:", self.spin_batch)
        hw_layout.addRow("Num Workers:", self.spin_workers)
        hw_group.setLayout(hw_layout)

        exec_group = QGroupBox("6. Execution")
        exec_layout = QVBoxLayout()
        exec_layout.setSpacing(10)
        
        seed_layout = QFormLayout()
        seed_layout.setSpacing(10)
        self.spin_seed = _SpinBox()
        self.spin_seed.setMinimumHeight(30)
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        seed_layout.addRow("Random Seed:", self.spin_seed)

        self.lbl_seed_scope = QLabel()
        self.lbl_seed_scope.setWordWrap(True)
        seed_layout.addRow("", self.lbl_seed_scope)
        
        self.btn_run = QPushButton("Run Pipeline")
        _set_role(self.btn_run, "primary")
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_run.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        
        self.btn_export = QPushButton("Export Results (CSV)")
        _set_role(self.btn_export, "neutral")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setVisible(False)

        exec_layout.addLayout(seed_layout)
        exec_layout.addWidget(self.btn_run)
        exec_layout.addWidget(self.progress_bar)
        exec_layout.addWidget(self.lbl_status)
        exec_layout.addWidget(self.btn_export)
        exec_group.setLayout(exec_layout)

        # Add groups to sidebar
        sidebar_layout.addWidget(data_group)
        sidebar_layout.addWidget(extract_group)
        sidebar_layout.addWidget(reduce_group)
        sidebar_layout.addWidget(cluster_group)
        sidebar_layout.addWidget(hw_group) # Re-added hardware group here
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(exec_group)
        # Annotation is a post-clustering step -> placed after the Run controls.
        sidebar_layout.addWidget(annot_group)

        # Right Panel: Expert Scientific Dashboard using QTabWidget
        self.tabs = QTabWidget()
        
        # Tab 1: 2D Projection Map
        self.tab_map = QWidget()
        tab_map_layout = QVBoxLayout(self.tab_map)
        tab_map_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_view = QWebEngineView()
        self.plot_view.setHtml("<div style='font-family: sans-serif; padding: 40px; text-align: center;'><h2>Latent Space Projection</h2><p>Run the pipeline to visualize the 2D structural embedding here.</p></div>")
        tab_map_layout.addWidget(self.plot_view)
        self.tabs.addTab(self.tab_map, "Projection Map")
        
        # Tab 2: Cluster Distribution
        self.tab_dist = QWidget()
        tab_dist_layout = QVBoxLayout(self.tab_dist)
        tab_dist_layout.setContentsMargins(0, 0, 0, 0)
        self.dist_view = QWebEngineView()
        self.dist_view.setHtml("<div style='font-family: sans-serif; padding: 40px; text-align: center;'><h2>Cluster Distribution</h2><p>Run the pipeline to see cluster assignments and class imbalances.</p></div>")
        tab_dist_layout.addWidget(self.dist_view)
        self.tabs.addTab(self.tab_dist, "Cluster Distribution")
        
        # Tab 3: Evaluation Metrics (Card Layout)
        self.tab_metrics = QWidget()
        self.metrics_layout = QVBoxLayout(self.tab_metrics)
        self.metrics_layout.setContentsMargins(30, 30, 30, 30)
        self.metrics_layout.setSpacing(20)
        
        self.metrics_title = QLabel("Evaluation Metrics")
        title_font2 = QFont()
        title_font2.setPointSize(18)
        title_font2.setBold(True)
        self.metrics_title.setFont(title_font2)
        self.metrics_layout.addWidget(self.metrics_title)
        
        self.metrics_desc = QLabel("Run the pipeline to generate metrics.")
        self.metrics_desc.setWordWrap(True)
        self.metrics_layout.addWidget(self.metrics_desc)
        
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(20)
        self.metrics_layout.addLayout(self.cards_layout)
        self.metrics_layout.addStretch()
        self.tabs.addTab(self.tab_metrics, "Evaluation Metrics")
        
        # Tab 4: Image Verification Gallery
        self.tab_gallery = QWidget()
        gallery_layout = QVBoxLayout(self.tab_gallery)
        gallery_layout.setContentsMargins(20, 20, 20, 20)
        
        # The label, the combo and the six action buttons need ~1110px side by
        # side. In a QHBoxLayout that meant the combo overlapped its own label
        # on any window narrower than ~1500px, so the selector gets its own
        # row and both rows wrap (FlowBar) instead of squeezing.
        selector_bar = FlowBar()
        lbl_select = QLabel("Select Cluster to Verify:")
        selector_bar.addWidget(lbl_select)
        self.combo_cluster_view = _ComboBox()
        self.combo_cluster_view.setMinimumWidth(200)
        self.combo_cluster_view.currentTextChanged.connect(self.update_gallery)
        selector_bar.addWidget(self.combo_cluster_view)
        gallery_layout.addWidget(selector_bar)

        gallery_controls = FlowBar()
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_decision)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self.redo_decision)
        self.btn_save_session = QPushButton("Save session")
        self.btn_save_session.clicked.connect(self.save_session)
        self.btn_load_session = QPushButton("Load session")
        self.btn_load_session.clicked.connect(self.load_session)
        self.btn_export_decisions = QPushButton("Export Decisions (CSV)")
        _set_role(self.btn_export_decisions, "neutral")
        self.btn_export_decisions.clicked.connect(self.export_decisions)
        self.btn_web_report = QPushButton("Generate Web Report")
        _set_role(self.btn_web_report, "accent")
        self.btn_web_report.clicked.connect(self.generate_web_report)
        for _b in (self.btn_undo, self.btn_redo, self.btn_save_session,
                   self.btn_load_session, self.btn_export_decisions, self.btn_web_report):
            gallery_controls.addWidget(_b)
        gallery_layout.addWidget(gallery_controls)

        # Tier filter bar: show only clusters of a given confidence tier.
        filter_bar = FlowBar()
        filter_bar.addWidget(QLabel("Filter:"))
        self._filter_buttons = {}
        for _key, _label in (("all", "All"), ("human_required", "Human Required"),
                             ("review", "Review"), ("auto_accept", "Auto-accepted"),
                             ("noise", "Noise")):
            _fb = QPushButton(_label)
            _fb.setCheckable(True)
            _fb.setProperty("seg", "true")
            _fb.clicked.connect(partial(self._set_tier_filter, _key))
            self._filter_buttons[_key] = _fb
            filter_bar.addWidget(_fb)
        self._filter_buttons["all"].setChecked(True)
        gallery_layout.addWidget(filter_bar)

        # Cluster-level corrections (act on the currently shown cluster).
        cluster_actions = FlowBar()
        self.btn_confirm_cluster = QPushButton("Confirm cluster (AI)")
        self.btn_confirm_cluster.clicked.connect(self.confirm_cluster)
        self.btn_confirm_runnerup = QPushButton("Confirm runner-up")
        self.btn_confirm_runnerup.clicked.connect(self.confirm_cluster_runnerup)
        self.combo_relabel = _ComboBox()
        self.combo_relabel.setEditable(True)
        self.combo_relabel.setMinimumWidth(200)
        self.btn_apply_relabel = QPushButton("Apply label to cluster")
        self.btn_apply_relabel.clicked.connect(self.relabel_cluster)
        self.btn_cluster_noise = QPushButton("All noise")
        self.btn_cluster_noise.clicked.connect(lambda: self.mark_cluster("noise"))
        self.btn_cluster_junk = QPushButton("All junk")
        self.btn_cluster_junk.clicked.connect(lambda: self.mark_cluster("junk"))
        for w in (self.btn_confirm_cluster, self.btn_confirm_runnerup, self.combo_relabel,
                  self.btn_apply_relabel, self.btn_cluster_noise, self.btn_cluster_junk):
            cluster_actions.addWidget(w)
        gallery_layout.addWidget(cluster_actions)
        self._set_correction_enabled(False)

        self.lbl_gallery_header = QLabel()
        self.lbl_gallery_header.setStyleSheet("font-weight: bold; font-size: 13px; color: #2f3542; padding: 4px 0;")
        gallery_layout.addWidget(self.lbl_gallery_header)

        # Cluster-level signal summary (matches the web review: cluster-mean
        # S/P/A/T + a cohesion-based Cluster conf). Per-image annotation
        # confidence stays on each card.
        self.lbl_cluster_signals = QLabel()
        self.lbl_cluster_signals.setStyleSheet(f"font-size: {T.TYPE_BODY}px; color: {T.INK_MUTED}; padding: 2px 0;")
        self.lbl_cluster_signals.setToolTip(
            "Cluster conf = cluster cohesion = 1 - (P*0.5 + A*0.5), from the "
            "cluster-mean spatial (P) and agreement (A) signals.\n"
            "This is separate from each image's annotation confidence "
            "(1 - combined uncertainty) shown on the cards.")
        gallery_layout.addWidget(self.lbl_cluster_signals)

        # Reference exemplars for the cluster's predicted species (top-1 +
        # runner-up), pulled from the dataset so the expert can compare.
        self.ref_container = QWidget()
        self.ref_layout = QHBoxLayout(self.ref_container)
        self.ref_layout.setContentsMargins(0, 4, 0, 4)
        self.ref_layout.setAlignment(Qt.AlignLeft)
        gallery_layout.addWidget(self.ref_container)

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_content = QWidget()
        self.gallery_grid = QGridLayout(self.gallery_content)
        self.gallery_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.gallery_grid.setSpacing(15)
        self.gallery_scroll.setWidget(self.gallery_content)
        gallery_layout.addWidget(self.gallery_scroll)
        self.tabs.addTab(self.tab_gallery, "Image Verification")

        # Tab 5: Run History. Leaderboard hero, compact list, details panel
        self.tab_history = QWidget()
        hist_root = QVBoxLayout(self.tab_history)
        hist_root.setContentsMargins(18, 18, 18, 18)
        hist_root.setSpacing(12)

        hist_header = QHBoxLayout()
        hist_title = QLabel("Run History")
        hist_title.setObjectName("PageTitle")
        hist_header.addWidget(hist_title)
        hist_header.addStretch()

        # Dataset filter + "group by config" toggle drive how the table is built.
        self._hist_loading = False   # guards against reload loops while populating
        hist_header.addWidget(QLabel("Dataset:"))
        self.hist_filter = _ComboBox()
        self.hist_filter.setMinimumWidth(160)
        self.hist_filter.currentTextChanged.connect(self._on_history_view_changed)
        hist_header.addWidget(self.hist_filter)
        self.hist_group_toggle = QCheckBox("Group by config (mean ± std)")
        self.hist_group_toggle.setChecked(True)
        self.hist_group_toggle.toggled.connect(self._on_history_view_changed)
        hist_header.addWidget(self.hist_group_toggle)

        self.btn_hist_refresh = QPushButton("Refresh")
        self.btn_hist_refresh.clicked.connect(self._load_history_table)
        self.btn_hist_open = QPushButton("Open folder")
        self.btn_hist_open.clicked.connect(self._history_open_folder)
        hist_header.addWidget(self.btn_hist_refresh)
        hist_header.addWidget(self.btn_hist_open)
        hist_root.addLayout(hist_header)

        # Summary strip: which configuration currently leads, and by what.
        # This used to be a saturated green slab headed "* BEST RUN" -- colour
        # and an icon doing the job of hierarchy. It sits on the normal card
        # surface now: the name reads first, the score is the single large
        # number, and the only colour is a status rule down the left edge.
        self.hist_hero = QFrame()
        self.hist_hero.setObjectName("Card")
        hero = QHBoxLayout(self.hist_hero)
        hero.setContentsMargins(20, 16, 20, 16)
        hero.setSpacing(28)

        hero_id = QVBoxLayout()
        hero_id.setSpacing(3)
        self.hero_eyebrow = QLabel("LEADING CONFIGURATION")
        self.hero_eyebrow.setObjectName("Eyebrow")
        self.hero_name = QLabel("No runs yet")
        self.hero_name.setObjectName("CardTitle")
        self.hero_name.setWordWrap(True)
        self.hero_sub = QLabel("Run the pipeline and every result is tracked here.")
        self.hero_sub.setObjectName("Caption")
        self.hero_sub.setWordWrap(True)
        hero_id.addWidget(self.hero_eyebrow)
        hero_id.addWidget(self.hero_name)
        hero_id.addWidget(self.hero_sub)
        hero.addLayout(hero_id, 1)

        def _hero_stat():
            """One right-aligned figure: small caps label over a large value."""
            col = QVBoxLayout()
            col.setSpacing(1)
            cap = QLabel("")
            cap.setObjectName("StatLabel")
            cap.setAlignment(Qt.AlignRight | Qt.AlignBottom)
            val = QLabel("—")
            val.setObjectName("StatValue")
            val.setAlignment(Qt.AlignRight | Qt.AlignTop)
            col.addStretch()
            col.addWidget(cap)
            col.addWidget(val)
            col.addStretch()
            return col, cap, val

        _c, self.hero_metric, self.hero_value = _hero_stat()
        hero.addLayout(_c)

        hero_rule = QFrame()
        hero_rule.setFrameShape(QFrame.VLine)
        hero_rule.setStyleSheet(f"color:{T.BORDER};")
        hero.addWidget(hero_rule)

        _c, self.hero_runs_cap, self.hero_runs_val = _hero_stat()
        self.hero_runs_cap.setText("RUNS")
        hero.addLayout(_c)

        _c, self.hero_cfg_cap, self.hero_cfg_val = _hero_stat()
        self.hero_cfg_cap.setText("CONFIGS")
        hero.addLayout(_c)

        hist_root.addWidget(self.hist_hero)

        # Leaderboard left, details right, on a splitter so the balance is
        # the reader's to set rather than a fixed ratio's.
        hist_split = QSplitter(Qt.Horizontal)
        hist_split.setHandleWidth(10)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Date", "Dataset", "Model", "Score"])
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)
        self.history_table.setSortingEnabled(True)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.itemSelectionChanged.connect(self._history_show_details)
        self.history_table.setMinimumWidth(420)
        hist_split.addWidget(self.history_table)

        details_panel = QFrame()
        details_panel.setObjectName("Card")
        dp_layout = QVBoxLayout(details_panel)
        dp_layout.setContentsMargins(16, 16, 16, 16)
        dp_layout.setSpacing(10)
        self.hist_details = QLabel("Select a run to see its full configuration and metrics.")
        self.hist_details.setTextFormat(Qt.RichText)
        self.hist_details.setWordWrap(True)
        self.hist_details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        dp_scroll = QScrollArea()
        dp_scroll.setWidgetResizable(True)
        dp_scroll.setFrameShape(QFrame.NoFrame)
        dp_scroll.setWidget(self.hist_details)
        dp_layout.addWidget(dp_scroll, 1)

        dp_btns = QHBoxLayout()
        self.btn_hist_apply = QPushButton("Apply settings")
        _set_role(self.btn_hist_apply, "accent")
        self.btn_hist_apply.clicked.connect(self._history_apply_selected)
        self.btn_hist_delete = QPushButton("Delete")
        self.btn_hist_delete.clicked.connect(self._history_delete_selected)
        dp_btns.addWidget(self.btn_hist_apply)
        dp_btns.addWidget(self.btn_hist_delete)
        dp_btns.addStretch()
        dp_layout.setSpacing(8)
        dp_layout.addLayout(dp_btns)
        # A 280px floor here vetoed the splitter entirely -- the handle would
        # not move. Kept low so dragging actually hands width to the table;
        # the panel cannot go below ~218px regardless, since its Apply/Delete
        # row sets an intrinsic minimum.
        details_panel.setMinimumWidth(200)
        hist_split.addWidget(details_panel)
        # Both of these must come AFTER addWidget: they index the splitter's
        # existing children. Called at construction, setCollapsible(1, ...)
        # warned "Index 1 out of range" and setChildrenCollapsible(False) did
        # not take either -- which left the TABLE draggable to zero width.
        hist_split.setCollapsible(0, False)   # never lose the leaderboard
        hist_split.setCollapsible(1, True)    # details may be squeezed away
        hist_split.setStretchFactor(0, 3)
        hist_split.setStretchFactor(1, 2)
        # The table carries seven columns; the details panel is scrollable
        # rich text and reads fine narrower. Weight the default accordingly.
        hist_split.setSizes([870, 280])

        hist_root.addWidget(hist_split, 1)
        self.tabs.addTab(self.tab_history, "Run History")
        self._load_history_table()

        # Wrap sidebar in a scroll area so dynamically added parameter widgets
        # never overflow the window height (e.g. umap adds 4+ rows to reduce_group)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(400)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)

        # Assemble main layout
        main_layout.addWidget(sidebar_scroll)
        main_layout.addWidget(self.tabs)

        # Initialize dynamic fields
        self._update_reduction_params(self.combo_reduction.currentText())
        self._update_cluster_params(self.combo_clustering.currentText())
        # Apply once at startup too -- the signal only fires on CHANGE, so
        # without this the pooling control stayed visible for the default
        # model (a BioCLIP, which cannot use it).
        self._update_pool_visibility()
        self._update_seed_scope()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _create_spinbox(self, val, min_v, max_v):
        sb = _SpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(val)
        sb.setMinimumHeight(30)
        return sb

    def _create_double_spinbox(self, val, min_v, max_v, step):
        sb = _DoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.setMinimumHeight(30)
        return sb
        
    def _create_combobox(self, items, default_idx=0):
        cb = _ComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(default_idx)
        cb.setMinimumHeight(30)
        return cb

    def _add_param(self, layout, key, label, widget, hint, store):
        """Add a labelled hyperparameter row with a hover hint on both the
        label and the input, then register the widget in `store`.

        The hint explains, in plain language, what happens when the user
        changes that hyperparameter."""
        lbl = QLabel(label)
        lbl.setToolTip(hint)
        widget.setToolTip(hint)
        layout.addRow(lbl, widget)
        store[key] = widget
        return widget

    def _update_reduction_params(self, method):
        self._clear_layout(self.reduction_params_layout)
        self.reduction_widgets.clear()
        
        if method == "none":
            return
            
        L = self.reduction_params_layout
        if "pca" in method:
            if method == "pca+umap":
                self._add_param(L, "pca_components",
                    "pca_components:", self._create_spinbox(100, 2, 768),
                    "PCA dimensions kept before UMAP. Higher preserves more "
                    "information but is slower and keeps more noise; lower "
                    "denoises and speeds up the following UMAP step.",
                    self.reduction_widgets)

            if method == "pca":
                self._add_param(L, "n_components",
                    "n_components:", self._create_spinbox(50, 2, 768),
                    "Final number of dimensions after PCA. Higher retains more "
                    "variance and detail; lower compresses harder and may drop "
                    "fine structure the clustering needs.",
                    self.reduction_widgets)

        if "umap" in method:
            self._add_param(L, "n_components",
                "n_components:", self._create_spinbox(50 if method=="pca+umap" else 100, 2, 768),
                "Number of dimensions UMAP projects to. Use 2 for the clearest "
                "visualization; higher can help clustering accuracy but adds "
                "noise and cannot be plotted directly.",
                self.reduction_widgets)

            self._add_param(L, "n_neighbors",
                "n_neighbors:", self._create_spinbox(15, 2, 200),
                "How many neighbours UMAP balances. Higher preserves global "
                "structure and gives broader clusters; lower emphasises local "
                "detail and can fragment groups into many small clusters.",
                self.reduction_widgets)

            self._add_param(L, "min_dist",
                "min_dist:", self._create_double_spinbox(0.1, 0.0, 1.0, 0.05),
                "Minimum spacing between points in the projection. Higher "
                "spreads points out evenly; lower packs them into tight, dense "
                "clumps, which usually makes clusters easier to separate.",
                self.reduction_widgets)

            self._add_param(L, "metric",
                "metric:", self._create_combobox(["cosine", "euclidean", "manhattan"]),
                "Distance measure for similarity. 'cosine' suits normalized "
                "embeddings (recommended for BioCLIP/DINO); 'euclidean' and "
                "'manhattan' also factor in vector magnitude.",
                self.reduction_widgets)

    def _update_pool_visibility(self):
        """Pooling only applies to the timm-loaded DINO models."""
        show = model_is_timm(self.combo_model.currentData())
        self.combo_pool.setVisible(show)
        self.lbl_pool.setVisible(show)

    def _update_cluster_params(self, method):
        self._clear_layout(self.clustering_params_layout)
        self.clustering_widgets.clear()
        
        L = self.clustering_params_layout
        if method in ["kmeans", "spectral", "agglomerative", "gmm"]:
            self._add_param(L, "n_clusters",
                "n_clusters:", self._create_spinbox(20, 2, 1000),
                "The number of clusters to create. Set it to how many groups "
                "(e.g. species) you expect. Too few merges distinct groups "
                "together; too many splits a single group apart.",
                self.clustering_widgets)

        if method == "kmeans":
            self._add_param(L, "n_init",
                "n_init:", self._create_spinbox(50, 1, 200),
                "Number of random restarts. Higher gives more stable, "
                "repeatable results but is slower; lower is faster but may "
                "settle on a poorer clustering.",
                self.clustering_widgets)

            self._add_param(L, "max_iter",
                "max_iter:", self._create_spinbox(300, 10, 2000),
                "Maximum refinement steps per restart. Higher lets the "
                "clustering fully converge; too low can stop before the "
                "cluster centres settle.",
                self.clustering_widgets)

        elif method == "spectral":
            self._add_param(L, "n_neighbors",
                "n_neighbors:", self._create_spinbox(15, 2, 200),
                "Neighbours used to build the similarity graph. Higher "
                "connects more points for smoother clusters; lower risks a "
                "disconnected graph that fragments the data.",
                self.clustering_widgets)

        elif method == "agglomerative":
            self._add_param(L, "linkage",
                "linkage:", self._create_combobox(["ward", "complete", "average", "single"]),
                "How the distance between clusters is measured when merging. "
                "'ward' gives compact, balanced clusters (recommended); "
                "'single' can chain points into long, stringy clusters.",
                self.clustering_widgets)

        elif method == "dbscan":
            self._add_param(L, "eps",
                "eps:", self._create_double_spinbox(0.5, 0.01, 10.0, 0.1),
                "Neighbourhood radius. Larger merges points into fewer, bigger "
                "clusters; smaller creates more clusters and labels more "
                "points as noise (outliers).",
                self.clustering_widgets)

            self._add_param(L, "min_samples",
                "min_samples:", self._create_spinbox(5, 1, 100),
                "Points required to form a dense core. Higher is stricter and "
                "marks more points as noise; lower is looser and rejects "
                "fewer points.",
                self.clustering_widgets)

        elif method == "hdbscan":
            self._add_param(L, "min_cluster_size",
                "min_cluster_size:", self._create_spinbox(5, 2, 100),
                "Smallest allowed cluster size. Higher merges small groups "
                "into fewer, larger clusters; lower keeps small clusters but "
                "labels more points as noise.",
                self.clustering_widgets)

            self._add_param(L, "min_samples",
                "min_samples:", self._create_spinbox(5, 1, 100),
                "How conservative the clustering is. Higher marks more points "
                "as noise and produces tighter cores; lower keeps more points "
                "and looser clusters.",
                self.clustering_widgets)

            self._add_param(L, "cluster_selection_method",
                "cluster_selection_method:", self._create_combobox(["eom", "leaf"]),
                "How clusters are pulled from the hierarchy. 'eom' (excess of "
                "mass) gives fewer, larger clusters (default); 'leaf' gives "
                "finer, more numerous clusters.",
                self.clustering_widgets)

        elif method == "gmm":
            self._add_param(L, "covariance_type",
                "covariance_type:", self._create_combobox(["full", "tied", "diag", "spherical"]),
                "The shape each cluster is allowed to take. 'full' fits "
                "arbitrary ellipses (most flexible, slower); 'diag'/'spherical' "
                "use simpler shapes that are faster but less flexible.",
                self.clustering_widgets)

            self._add_param(L, "max_iter",
                "max_iter:", self._create_spinbox(200, 10, 2000),
                "Maximum EM fitting iterations. Higher gives the model more "
                "converges; too low can stop before the fit stabilises.",
                self.clustering_widgets)

    _SEED_AWARE_CLUSTERING = {"kmeans", "spectral", "gmm"}

    def _update_seed_scope(self, *_):
        reduction = self.combo_reduction.currentText()
        clustering = self.combo_clustering.currentText()
        uses_reduction = reduction != "none"
        uses_clustering = clustering in self._SEED_AWARE_CLUSTERING

        if uses_reduction and uses_clustering:
            text = "Affects: reduction + clustering"
            color = "#2ed573"
        elif uses_reduction:
            text = f"Affects: reduction only ({clustering} is deterministic)"
            color = "#ffa502"
        elif uses_clustering:
            text = "Affects: clustering only"
            color = "#2ed573"
        else:
            text = f"Not used ({clustering} is deterministic, no reduction)"
            color = "#a4b0be"

        self.lbl_seed_scope.setText(text)
        self.lbl_seed_scope.setStyleSheet(f"color: {color}; font-style: italic; font-size: 11px;")

    def _set_sidebar_enabled(self, enabled: bool):
        for w in (self.btn_browse, self.chk_labels, self.combo_model,
                  self.combo_reduction, self.combo_clustering,
                  self.combo_device, self.spin_batch, self.spin_workers,
                  self.spin_seed):
            w.setEnabled(enabled)
        for w in self.reduction_widgets.values():
            w.setEnabled(enabled)
        for w in self.clustering_widgets.values():
            w.setEnabled(enabled)
        if enabled:
            self.btn_run.setEnabled(True)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_path = folder
            self.lbl_folder.setText(folder)
            self.btn_run.setEnabled(True)

    # -------------------- Annotation --------------------

    def _update_candidate_source(self, *_):
        is_csv = self.combo_candidate.currentData() == "csv"
        self.btn_browse_csv.setVisible(is_csv)
        self.lbl_csv.setVisible(is_csv)

    def browse_taxonomy_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Taxonomy CSV", "", "CSV Files (*.csv)")
        if path:
            self.taxonomy_csv_path = path
            self.lbl_csv.setText(os.path.basename(path))

    def run_annotation_step(self):
        if not hasattr(self, "current_result"):
            return
        model_key = self.combo_model.currentData()
        if not model_has_text(model_key):
            QMessageBox.warning(self, "Annotation unavailable",
                "Annotation needs a model with a text encoder. "
                "DINOv2 and DINOv3 are vision-only — re-run with a BioCLIP "
                "or CLIP model.")
            return

        source = self.combo_candidate.currentData()
        folder_class_names = None
        if source == "csv":
            if not self.taxonomy_csv_path:
                QMessageBox.warning(self, "No CSV selected",
                    "Choose a taxonomy CSV (columns: class_name, genus, family, order).")
                return
        else:  # folder labels
            if self.current_result.get("class_names") is None:
                QMessageBox.warning(self, "No labels available",
                    "Folder-label candidates require the 'Folder has Ground Truth "
                    "Labels' option to be checked before running. Otherwise choose "
                    "a taxonomy CSV.")
                return
            folder_class_names = list(self.current_result["class_names"])

        # Configurable uncertainty weights (keys match run_annotation).
        weights = {
            "text_margin":             self.spin_w_sem.value(),
            "membership_probability":  self.spin_w_spa.value(),
            "cluster_agreement":       self.spin_w_agr.value(),
            "taxonomic_consistency":   self.spin_w_tax.value(),
        }
        if sum(weights.values()) <= 0:
            QMessageBox.warning(self, "Invalid weights",
                "At least one uncertainty weight must be greater than 0.")
            return

        # Reuse the loaded model only if it matches the selected one.
        reuse_extractor = None
        if (self._cached_extractor is not None
                and self._cached_extractor_model == model_key):
            reuse_extractor = self._cached_extractor

        self.btn_annotate.setEnabled(False)
        self.lbl_annot_status.setText("Starting annotation...")
        self.annot_worker = AnnotationWorker(
            features=self.current_result["features"],       # ORIGINAL features
            cluster_ids=self.current_result["cluster_ids"],
            image_paths=self.current_result["image_paths"],
            candidate_source=source,
            extractor_name=model_key,
            device=self.combo_device.currentData(),
            batch_size=self.spin_batch.value(),
            weights=weights,
            review_threshold=self.spin_rev.value(),
            human_threshold=self.spin_hum.value(),
            taxonomy_csv=self.taxonomy_csv_path,
            folder_class_names=folder_class_names,
            probabilities=self.current_result.get("probabilities"),
            extractor=reuse_extractor,
        )
        self.annot_worker.progress.connect(self._annot_progress)
        self.annot_worker.finished.connect(self._annot_finished)
        self.annot_worker.error.connect(self._annot_error)
        self.annot_worker.start()

    def _annot_progress(self, val, msg):
        self.lbl_annot_status.setText(msg)

    def _annot_finished(self, df):
        self.annotation_df = df
        self._ann_lookup = {str(r["file_path"]): r for _, r in df.iterrows()}
        # Seed one decision per image from the AI result (noise clusters -> noise).
        self.decisions = {}
        for _, r in df.iterrows():
            p = str(r["file_path"])
            self.decisions[p] = {
                "final": str(r["ai_annotation"]),
                "status": "noise" if int(r["cluster_id"]) == -1 else "species",
            }
        # Candidate species for the relabel dropdowns.
        cands = set()
        for c in ("ai_annotation", "ai_second_choice"):
            if c in df.columns:
                cands.update(str(x) for x in df[c].dropna().unique()
                             if str(x) and str(x).lower() != "nan")
        self._candidates = sorted(cands)
        self.combo_relabel.clear()
        self.combo_relabel.addItems(self._candidates)

        # Worst per-image tier for each cluster (drives the filter + ordering).
        order = {"human_required": 0, "review": 1, "auto_accept": 2}
        self._cluster_tier = {}
        for cid, grp in df.groupby("cluster_id"):
            if int(cid) == -1:
                continue
            worst = min((str(t) for t in grp["confidence_tier"]),
                        key=lambda t: order.get(t, 3))
            self._cluster_tier[str(int(cid))] = worst

        # Reference images per species (dataset folders), keyed by the same
        # class name the annotator uses as ai_annotation.
        self._ref_map = {}
        tl = self.current_result.get("true_labels")
        cn = self.current_result.get("class_names")
        if tl is not None and cn is not None:
            ip = self.current_result["image_paths"]
            rm = {}
            for i, l in enumerate(tl):
                rm.setdefault(str(cn[int(l)]), []).append(ip[i])
            self._ref_map = rm

        # Annotation accuracy vs ground truth (only when folders are labeled).
        # Correct = the AI-annotated species matches the image's true class.
        self._img_correct = {}
        self._overall_acc = None
        self._cluster_acc = None
        if tl is not None and cn is not None:
            from collections import Counter as _Ctr
            def _nrm(s):
                return str(s).lower().replace("_", " ").strip()
            true_of = {str(ip[i]): _nrm(cn[int(tl[i])]) for i in range(len(tl))}
            # Per-image accuracy: AI species == that image's true class.
            correct = total = 0
            for _, r in df.iterrows():
                p = str(r["file_path"])
                ai = str(r["ai_annotation"])
                if not ai or int(r["cluster_id"]) == -1 or p not in true_of:
                    continue
                ok = (_nrm(ai) == true_of[p])
                self._img_correct[p] = ok
                correct += int(ok)
                total += 1
            self._overall_acc = (correct / total) if total else None
            self._overall_acc_n = (correct, total)
            # Cluster-naming accuracy: a cluster is correct if its AI label
            # matches the majority TRUE class of its images.
            c_ok = c_tot = 0
            for cid, grp in df[df["cluster_id"] != -1].groupby("cluster_id"):
                truths = [true_of[str(p)] for p in grp["file_path"] if str(p) in true_of]
                if not truths:
                    continue
                ai = _nrm(str(grp["ai_annotation"].iloc[0]))
                maj = _Ctr(truths).most_common(1)[0][0]
                c_ok += int(ai == maj)
                c_tot += 1
            self._cluster_acc = (c_ok / c_tot) if c_tot else None
            self._cluster_acc_n = (c_ok, c_tot)

        # Undo/redo history starts fresh; set the per-dataset autosave file.
        self._undo, self._redo = [], []
        import hashlib
        key = str(self.folder_path or "dataset")
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        self._autosave_path = os.path.join(
            os.path.expanduser("~"), "BioCluster", f"autosave_{h}.json")

        self._set_correction_enabled(True)
        self.btn_annotate.setEnabled(True)

        # Offer to restore autosaved corrections for this dataset.
        if os.path.exists(self._autosave_path):
            if QMessageBox.question(
                    self, "Restore corrections?",
                    "Found autosaved corrections for this dataset. Restore them?",
                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                try:
                    with open(self._autosave_path, encoding="utf-8") as f:
                        saved = json.load(f)
                    self.decisions.update({str(k): v for k, v in saved.items()})
                except Exception:
                    pass

        n_auto = int((df["confidence_tier"] == "auto_accept").sum())
        n_rev  = int((df["confidence_tier"] == "review").sum())
        n_hum  = int((df["confidence_tier"] == "human_required").sum())
        self.lbl_annot_status.setText(
            f"Done: {n_auto} auto · {n_rev} review · {n_hum} human"
            + (f"  ·  img acc {self._overall_acc*100:.1f}% "
               f"({self._overall_acc_n[0]}/{self._overall_acc_n[1]})"
               if self._overall_acc is not None else "")
            + (f"  ·  cluster acc {self._cluster_acc*100:.1f}% "
               f"({self._cluster_acc_n[0]}/{self._cluster_acc_n[1]})"
               if self._cluster_acc is not None else ""))
        self._tier_filter = "all"
        self._populate_cluster_dropdown()   # worst-first, overlays appear
        self.tabs.setCurrentIndex(3)   # Image Verification tab

    def _annot_error(self, msg, tb):
        self.btn_annotate.setEnabled(True)
        self.lbl_annot_status.setText("Annotation error.")
        m = QMessageBox(self)
        m.setIcon(QMessageBox.Critical)
        m.setWindowTitle("Annotation Error")
        m.setText(msg)
        m.setDetailedText(tb)
        m.exec()

    def run_pipeline(self):
        if not self.folder_path:
            return

        self._set_sidebar_enabled(False)
        self.btn_export.setVisible(False)
        self.btn_run.setText("Cancel")
        _set_role(self.btn_run, "danger")
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Initializing...")

        # Collect reduction params
        reduction_params = {}
        for key, widget in self.reduction_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                reduction_params[key] = widget.value()
            elif isinstance(widget, QComboBox):
                reduction_params[key] = widget.currentText()
        reduction_params["random_state"] = self.spin_seed.value()
                
        # Collect clustering params
        clustering_params = {}
        for key, widget in self.clustering_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                clustering_params[key] = widget.value()
            elif isinstance(widget, QComboBox):
                clustering_params[key] = widget.currentText()
        clustering_params["random_state"] = self.spin_seed.value()

        # Decide whether we can reuse the already-loaded model. The key is
        # (model, device): same key -> reuse (no reload); different key ->
        # free the old model NOW so the new one loads into clean VRAM.
        model_key = (self.combo_model.currentData(), self.combo_device.currentData())
        shared_extractor = None
        if self._cached_extractor is not None and self._cached_extractor_key == model_key:
            shared_extractor = self._cached_extractor
        elif self._cached_extractor is not None:
            # Switching models: release the previous one before loading.
            self._cached_extractor = None
            self._cached_extractor_key = None
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self.worker = PipelineWorker(
            folder=self.folder_path,
            has_labels=self.chk_labels.isChecked(),
            extractor_name=self.combo_model.currentData(),
            reduction_method=self.combo_reduction.currentText(),
            reduction_params=reduction_params,
            clustering_method=self.combo_clustering.currentText(),
            clustering_params=clustering_params,
            device=self.combo_device.currentData(),
            batch_size=self.spin_batch.value(),
            num_workers=self.spin_workers.value(),
            shared_extractor=shared_extractor,
            model_key=model_key,
            global_pool=self.combo_pool.currentData() or None,
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.pipeline_finished)
        self.worker.error.connect(self.pipeline_error)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        self.worker.start()
        self.btn_run.clicked.disconnect()
        self.btn_run.clicked.connect(self.cancel_pipeline)

    def _reset_run_button(self):
        self._set_sidebar_enabled(True)
        self.btn_run.setText("Run Pipeline")
        _set_role(self.btn_run, "primary")
        self.btn_run.clicked.disconnect()
        self.btn_run.clicked.connect(self.run_pipeline)

    def cancel_pipeline(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.lbl_status.setText("Cancelling...")
            self.worker.requestInterruption()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def _log_run_history(self, result, n_images, n_clusters):
        """Append this run's settings + metrics to the persistent history CSV
        so the user never loses track of what config produced what result."""
        import json, datetime
        try:
            w = getattr(self, "worker", None)
            metrics = result.get("metrics", {}) or {}
            cluster_ids = np.asarray(result["cluster_ids"])
            noise_pct = round(float((cluster_ids == -1).mean()) * 100, 2)
            rp = dict(getattr(w, "reduction_params", {}) or {})
            cp = dict(getattr(w, "clustering_params", {}) or {})
            seed = rp.get("random_state", cp.get("random_state", ""))
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "dataset": os.path.basename(self.folder_path or ""),
                "folder": self.folder_path or "",
                "n_images": n_images,
                "n_clusters": n_clusters,
                "noise_pct": noise_pct,
                "model": getattr(w, "extractor_name", ""),
                "device": getattr(w, "device", ""),
                "labeled": getattr(w, "has_labels", ""),
                # Recorded because they perturb the embeddings at float level
                # even though they do not change the computation.
                "batch_size": getattr(w, "batch_size", ""),
                "num_workers": getattr(w, "num_workers", ""),
                "reduction": getattr(w, "reduction_method", ""),
                "reduction_params": json.dumps({k: v for k, v in rp.items() if k != "random_state"}),
                "clustering": getattr(w, "clustering_method", ""),
                "clustering_params": json.dumps({k: v for k, v in cp.items() if k != "random_state"}),
                "seed": seed,
                "ARI": metrics.get("ARI", ""),
                "NMI": metrics.get("NMI", ""),
                "Hungarian": metrics.get("Hungarian Accuracy", ""),
                "Silhouette": metrics.get("Silhouette Score", ""),
                "Calinski_Harabasz": metrics.get("Calinski-Harabasz", ""),
                "Davies_Bouldin": metrics.get("Davies-Bouldin", ""),
            }
            write_run_history(row)
        except Exception as e:
            print(f"[history] could not log run: {e}")

    # ---- Run History tab -------------------------------------------------
    @staticmethod
    def _fnum(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fmt_ts(ts):
        ts = str(ts)
        return ts.replace("T", " ")[:16] if "T" in ts else ts

    def _row_score(self, row):
        """Primary score for a run: ARI if labeled, else Silhouette.
        Returns (float_or_None, display_text, metric_name)."""
        ari = self._fnum(row.get("ARI"))
        if ari is not None:
            return ari, f"{ari:.3f}", "ARI"
        sil = self._fnum(row.get("Silhouette"))
        if sil is not None:
            return sil, f"{sil:.3f}", "Silhouette"
        return None, "—", "score"

    @staticmethod
    def _score_color(val):
        """Reserved status colour for a primary score.

        Thresholds and colours both come from theme.status_for, so a green
        score in this table means exactly what a green metric card means.
        """
        if val is None:
            return T.INK_MUTED
        return T.status_for("ARI", val) or T.INK_SECONDARY

    @staticmethod
    def _pipeline_str(row):
        return f"{row.get('reduction', '?')} → {row.get('clustering', '?')}"

    def _pipeline_tooltip(self, row):
        """Full parameters for a run, for the Pipeline cell's tooltip. The
        column itself stays at method level -- the detail is one hover away
        and set out properly in the panel beside the table."""
        def fmt(js):
            try:
                d = json.loads(js or "{}")
            except Exception:
                d = {}
            return ", ".join(f"{k}={v}" for k, v in d.items()) or "default"

        return (f"Reduction: {row.get('reduction', '?')} "
                f"({fmt(row.get('reduction_params'))})\n"
                f"Clustering: {row.get('clustering', '?')} "
                f"({fmt(row.get('clustering_params'))})")

    def _on_history_view_changed(self, *_):
        """Filter dropdown / group toggle changed -> rebuild (unless we are the
        ones repopulating the dropdown)."""
        if not getattr(self, "_hist_loading", False):
            self._load_history_table()

    def _load_history_table(self):
        """(Re)load run_history.csv into the leaderboard, honouring the dataset
        filter and the 'group by config' toggle. Column 0 of each table row
        carries its backing object: a run dict, or a config-group dict in
        grouped mode, so Apply/Delete keep working after sorting."""
        import csv
        tbl = self.history_table
        tbl.setSortingEnabled(False)
        tbl.clearContents()
        tbl.setRowCount(0)
        self._history_rows = []

        if not os.path.exists(_HISTORY_CSV):
            self._update_hero(None)
            self.hist_details.setText("Select a run to see its full configuration and metrics.")
            self._populate_dataset_filter([])
            tbl.setSortingEnabled(True)
            return
        try:
            with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"[history] read failed: {e}")
            tbl.setSortingEnabled(True)
            return
        # Stable per-row id. A dict stored in Qt.UserRole round-trips through
        # QVariantMap, so data() hands back an equal-but-distinct dict --
        # identity (`is` / `id()`) never matches the object we stored. Rows
        # therefore carry an explicit uid that survives the conversion. It is
        # not a _HISTORY_COLUMNS field, so it is never written to the CSV.
        for i, r in enumerate(rows):
            r["_uid"] = i
        self._history_rows = rows

        self._populate_dataset_filter(rows)
        sel = self.hist_filter.currentText()
        view = rows if sel in ("", "All") else [r for r in rows if r.get("dataset") == sel]

        best = self._best_history_row(view)
        self._update_hero(best)

        grouped = self.hist_group_toggle.isChecked()
        if grouped:
            self._populate_grouped(view)
        else:
            self._populate_runs(view, best)

        # Order is applied explicitly, then the selection is placed on top of
        # the settled order. Doing it the other way round moved the selection.
        tbl.setSortingEnabled(True)
        if grouped:
            tbl.sortItems(0, Qt.AscendingOrder)     # by rank, best first
            self._size_history_columns()
            if tbl.rowCount():
                tbl.selectRow(0)
        else:
            tbl.sortItems(0, Qt.DescendingOrder)    # by date, newest first
            self._size_history_columns()
            self._select_history_row(best)          # uid-based, survives sorting

    def _populate_dataset_filter(self, rows):
        """Fill the dataset dropdown with the distinct dataset names, preserving
        the current selection. Guarded so it doesn't trigger a reload loop."""
        self._hist_loading = True
        try:
            cur = self.hist_filter.currentText()
            datasets = sorted({r.get("dataset", "") for r in rows if r.get("dataset")})
            items = ["All"] + datasets
            self.hist_filter.clear()
            self.hist_filter.addItems(items)
            if cur in items:
                self.hist_filter.setCurrentText(cur)
        finally:
            self._hist_loading = False

    def _populate_runs(self, view, best):
        """Per-run view: every run as its own row."""
        tbl = self.history_table
        tbl.setColumnCount(7)
        tbl.setHorizontalHeaderLabels(["Date", "Dataset", "Model", "Pipeline",
                                       "Score", "Clusters", "Noise %"])
        tbl.setRowCount(len(view))
        for r, row in enumerate(reversed(view)):   # newest at top
            it0 = QTableWidgetItem(self._fmt_ts(row.get("timestamp", "")))
            it0.setData(Qt.UserRole, row)
            tbl.setItem(r, 0, it0)
            dit = QTableWidgetItem(str(row.get("dataset", "")))
            dit.setToolTip(str(row.get("dataset", "")))
            tbl.setItem(r, 1, dit)
            tbl.setItem(r, 2, QTableWidgetItem(str(row.get("model", ""))))
            pit = QTableWidgetItem(self._pipeline_str(row))
            pit.setToolTip(self._pipeline_tooltip(row))
            tbl.setItem(r, 3, pit)
            sval, stext, _ = self._row_score(row)
            sit = QTableWidgetItem()
            sit.setData(Qt.DisplayRole, round(sval, 4) if sval is not None else stext)
            sit.setForeground(QColor(self._score_color(sval)))
            sf = sit.font(); sf.setBold(True); sit.setFont(sf)
            tbl.setItem(r, 4, sit)

            # Numeric DisplayRole so the columns sort as numbers, not strings.
            nc = self._fnum(row.get("n_clusters"))
            cit = QTableWidgetItem()
            cit.setData(Qt.DisplayRole, int(nc) if nc is not None else "—")
            tbl.setItem(r, 5, cit)

            npct = self._fnum(row.get("noise_pct"))
            nit = QTableWidgetItem()
            nit.setData(Qt.DisplayRole, round(npct, 2) if npct is not None else "—")
            if npct is not None and npct >= 10:
                # High noise inflates ARI, which is scored on kept points only.
                nit.setForeground(QColor(T.STATUS_WARNING))
                nf = nit.font(); nf.setBold(True); nit.setFont(nf)
                nit.setToolTip(f"{npct:.1f}% of images were labelled noise and are "
                               f"excluded from ARI/NMI/Hungarian — the score "
                               f"describes only the remaining images.")
            tbl.setItem(r, 6, nit)

            if row is best:
                # Rows here are newest-first, so the best one needs marking.
                # Weight rather than a colour wash: the wash sat on top of the
                # alternating stripes and read as a selection.
                for c in range(tbl.columnCount()):
                    it = tbl.item(r, c)
                    if it:
                        bf = it.font(); bf.setBold(True); it.setFont(bf)

    # Columns holding a number. They align right so digits line up down the
    # column; everything else is text and aligns left. "#" is a rank label,
    # not a quantity, so it centres.
    _NUM_COLS = {"Score", "n", "Clusters", "Noise %"}
    _MID_COLS = {"#"}

    def _size_history_columns(self):
        """Size every column to its own content and align header to cells.

        No column stretches. Stretching makes one column absorb every
        shortfall, and whichever one that is becomes unreadable first -- which
        is how Pipeline ended up showing "uma..." and then Dataset ended up
        elided in its place. Sizing all of them to their content means nothing
        is ever cut; when the window is too narrow for the total, the table
        scrolls horizontally, which is recoverable in a way that truncation is
        not.
        """
        tbl = self.history_table
        hh = tbl.horizontalHeader()
        hh.setStretchLastSection(False)
        for c in range(tbl.columnCount()):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        tbl.setWordWrap(False)

        # Header and cells must agree, or the column reads as ragged: the
        # headers were centred over left-aligned text and left-aligned numbers.
        for c in range(tbl.columnCount()):
            head = tbl.horizontalHeaderItem(c)
            name = head.text() if head else ""
            if name in self._MID_COLS:
                align = Qt.AlignCenter
            elif name in self._NUM_COLS:
                align = Qt.AlignRight | Qt.AlignVCenter
            else:
                align = Qt.AlignLeft | Qt.AlignVCenter
            if head:
                head.setTextAlignment(align)
            for r in range(tbl.rowCount()):
                it = tbl.item(r, c)
                if it:
                    it.setTextAlignment(align)

    def _config_key(self, row):
        """Config identity WITHOUT the seed, so seed-replicas group together."""
        return _run_identity(row)[:-1]

    def _group_by_config(self, view):
        """Collapse runs into one entry per configuration, with mean/std of the
        primary score. Ranked best-first."""
        import statistics
        from collections import defaultdict
        groups = defaultdict(list)
        for r in view:
            groups[self._config_key(r)].append(r)
        out = []
        for _key, grp in groups.items():
            scores = [s for s in (self._row_score(r)[0] for r in grp) if s is not None]
            mean = sum(scores) / len(scores) if scores else None
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            rep = max(grp, key=lambda r: (self._row_score(r)[0]
                                          if self._row_score(r)[0] is not None else float("-inf")))
            out.append({"_group": True, "runs": grp, "rep": rep,
                        "mean": mean, "std": std, "n": len(grp)})
        out.sort(key=lambda g: (g["mean"] if g["mean"] is not None else float("-inf")), reverse=True)
        return out

    def _populate_grouped(self, view):
        """Grouped view: one row per configuration, score = mean ± std (n)."""
        tbl = self.history_table
        groups = self._group_by_config(view)
        tbl.setColumnCount(8)
        tbl.setHorizontalHeaderLabels(["#", "Dataset", "Model", "Pipeline",
                                       "Score", "n", "Clusters", "Noise %"])
        for _c, _tip in ((0, "Rank by mean score, best first."),
                         (3, "Reduction and clustering method. Hover a cell "
                             "for the full parameters."),
                         (4, "Mean ± standard deviation of the primary score "
                             "(ARI when labelled, otherwise Silhouette) across "
                             "the seeds in this configuration."),
                         (5, "Number of seed replicates in this configuration.")):
            _h = tbl.horizontalHeaderItem(_c)
            if _h:
                _h.setToolTip(_tip)
        tbl.setRowCount(len(groups))
        for r, g in enumerate(groups):
            rep = g["rep"]
            # Groups arrive ranked best-first. A rank number says "leading"
            # without a star or a coloured row, and it survives re-sorting on
            # any other column -- which a highlight on one row does not.
            it0 = QTableWidgetItem()
            it0.setData(Qt.DisplayRole, r + 1)
            it0.setData(Qt.UserRole, g)
            if r == 0:
                f0 = it0.font(); f0.setBold(True); it0.setFont(f0)
            tbl.setItem(r, 0, it0)
            dit = QTableWidgetItem(str(rep.get("dataset", "")))
            dit.setToolTip(str(rep.get("dataset", "")))
            tbl.setItem(r, 1, dit)
            tbl.setItem(r, 2, QTableWidgetItem(str(rep.get("model", ""))))
            pit = QTableWidgetItem(self._pipeline_str(rep))
            pit.setToolTip(self._pipeline_tooltip(rep))
            tbl.setItem(r, 3, pit)
            mean = g["mean"]
            sit = QTableWidgetItem()
            sit.setData(Qt.DisplayRole,
                        f"{mean:.3f} ± {g['std']:.3f}" if mean is not None else "—")
            sit.setForeground(QColor(self._score_color(mean)))
            sf = sit.font(); sf.setBold(True); sit.setFont(sf)
            tbl.setItem(r, 4, sit)
            nit_n = QTableWidgetItem()
            nit_n.setData(Qt.DisplayRole, g["n"])
            tbl.setItem(r, 5, nit_n)

            def _avg(key):
                vals = [v for v in (self._fnum(x.get(key)) for x in g["runs"])
                        if v is not None]
                return sum(vals) / len(vals) if vals else None

            mc, mn = _avg("n_clusters"), _avg("noise_pct")
            cit = QTableWidgetItem()
            cit.setData(Qt.DisplayRole, round(mc, 1) if mc is not None else "—")
            tbl.setItem(r, 6, cit)
            nit = QTableWidgetItem()
            nit.setData(Qt.DisplayRole, round(mn, 2) if mn is not None else "—")
            if mn is not None and mn >= 10:
                nit.setForeground(QColor(T.STATUS_WARNING))
                nf = nit.font(); nf.setBold(True); nit.setFont(nf)
            tbl.setItem(r, 7, nit)


    def _set_hero_rule(self, colour):
        """Status rule down the card's left edge. The card's own background and
        border still come from the global sheet -- only the rule is local."""
        self.hist_hero.setStyleSheet(
            f"QFrame#Card{{border-left:3px solid {colour};}}" if colour else "")

    def _update_hero(self, best):
        if not best:
            self.hero_name.setText("No runs yet")
            self.hero_sub.setText("Run the pipeline and every result is tracked here.")
            self.hero_value.setText("—")
            self.hero_metric.setText("")
            self.hero_runs_val.setText("0")
            self.hero_cfg_val.setText("0")
            self._set_hero_rule(None)
            return

        score, stext, metric = self._row_score(best)
        n = len(self._history_rows)
        n_cfg = len({self._config_key(r) for r in self._history_rows})

        self.hero_name.setText(
            f"{best.get('model', '')} · {self._pipeline_str(best)}")
        clusters = best.get("n_clusters", "?")
        self.hero_sub.setText(f"{best.get('dataset', '')} · {clusters} clusters")
        self.hero_metric.setText(metric.upper())
        self.hero_value.setText(stext)
        self.hero_value.setStyleSheet(f"color:{self._score_color(score)};")
        self.hero_runs_val.setText(str(n))
        self.hero_cfg_val.setText(str(n_cfg))
        self._set_hero_rule(self._score_color(score))

    def _select_history_row(self, row):
        """Select the table row backed by `row` (falls back to the first row)."""
        tbl = self.history_table
        uid = row.get("_uid") if isinstance(row, dict) else None
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            if it is None:
                continue
            obj = it.data(Qt.UserRole)
            # Match on uid, not identity: Qt returns a copy of the stored dict.
            if uid is not None and isinstance(obj, dict) and obj.get("_uid") == uid:
                tbl.selectRow(r)
                return
        if tbl.rowCount() > 0:
            tbl.selectRow(0)

    def _history_show_details(self):
        obj = self._history_selected_row()
        if not obj:
            return
        if isinstance(obj, dict) and obj.get("_group"):
            self.hist_details.setText(self._config_details_html(obj))
        else:
            self.hist_details.setText(self._details_html(obj))

    def _config_details_html(self, g):
        """Details for a grouped configuration: params, per-metric mean ± std,
        and a per-seed breakdown (drill-down)."""
        import json
        import statistics
        rep = g["rep"]

        def params(s):
            try:
                d = json.loads(s or "{}")
            except Exception:
                d = {}
            return ", ".join(f"{k}={v}" for k, v in d.items()) or "default"

        def line(label, value):
            value = value if value not in ("", None) else "—"
            return (f"<tr><td style='padding:3px 12px 3px 0; opacity:0.75;'>{label}</td>"
                    f"<td><b>{value}</b></td></tr>")

        mrows = ""
        for m, label in [("ARI", "ARI"), ("NMI", "NMI"), ("Hungarian", "Hungarian"),
                         ("Silhouette", "Silhouette"),
                         ("Calinski_Harabasz", "Calinski-Harabasz"),
                         ("Davies_Bouldin", "Davies-Bouldin")]:
            vals = [v for v in (self._fnum(r.get(m)) for r in g["runs"]) if v is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mrows += line(label, f"{mean:.4f} ± {std:.4f}  (n={len(vals)})")

        # Cluster structure. For density methods (hdbscan/dbscan) these are
        # PRIMARY results, not diagnostics: ARI is computed on non-noise points
        # only, so a run that discards half the images as noise scores highly on
        # the easy remainder. Ranking on ARI alone hides that -- show it here.
        srows = ""
        for key, label, fmt in [("n_clusters", "Clusters found", "{:.1f}"),
                                ("noise_pct",  "Noise %",        "{:.2f}")]:
            vals = [v for v in (self._fnum(r.get(key)) for r in g["runs"]) if v is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            extra = ""
            if key == "noise_pct" and max(vals) >= 10:
                extra = (f" &nbsp;<span style='color:{T.STATUS_WARNING};'>"
                         f"max {max(vals):.1f}% — ARI ignores noise points</span>")
            srows += line(label, fmt.format(mean) + f" ± {fmt.format(std)}"
                          f"  (min {min(vals):g}, max {max(vals):g})" + extra)

        seed_rows = ""
        for r in sorted(g["runs"], key=lambda r: str(r.get("seed", ""))):
            n_c = r.get("n_clusters", "?")
            noise = r.get("noise_pct", "")
            suffix = f"  ({n_c} clusters, {noise}% noise)" if n_c not in ("", None) else ""
            seed_rows += line(f"seed {r.get('seed', '?')}",
                              self._row_score(r)[1] + suffix)

        return (
            f"<div style='font-size:15px; font-weight:bold; padding-bottom:6px;'>"
            f"{rep.get('model', '')} &nbsp;·&nbsp; {self._pipeline_str(rep)}</div>"
            f"<div style='opacity:0.7; padding-bottom:10px;'>{rep.get('dataset', '')}"
            f" &nbsp;·&nbsp; {g['n']} run(s)</div>"
            f"<table cellspacing='0'>"
            + line("Reduction", f"{rep.get('reduction', '')} ({params(rep.get('reduction_params'))})")
            + line("Clustering", f"{rep.get('clustering', '')} ({params(rep.get('clustering_params'))})")
            + line("Device", rep.get("device"))
            + "</table>"
            + "<div style='font-weight:bold; padding:12px 0 4px 0;'>Metrics (mean ± std)</div>"
            + f"<table cellspacing='0'>{mrows}</table>"
            + "<div style='font-weight:bold; padding:12px 0 4px 0;'>Cluster structure (mean ± std)</div>"
            + f"<table cellspacing='0'>{srows}</table>"
            + "<div style='font-weight:bold; padding:12px 0 4px 0;'>Runs by seed</div>"
            + f"<table cellspacing='0'>{seed_rows}</table>")

    def _details_html(self, row):
        """Readable rich-text breakdown of one run (replaces raw CSV columns)."""
        import json

        def params(s):
            try:
                d = json.loads(s or "{}")
            except Exception:
                d = {}
            return ", ".join(f"{k}={v}" for k, v in d.items()) or "default"

        def line(label, value):
            value = value if value not in ("", None) else "—"
            return (f"<tr><td style='padding:3px 12px 3px 0; opacity:0.75;'>{label}</td>"
                    f"<td><b>{value}</b></td></tr>")

        metrics_rows = "".join([
            line("ARI", row.get("ARI")),
            line("NMI", row.get("NMI")),
            line("Hungarian", row.get("Hungarian")),
            line("Silhouette", row.get("Silhouette")),
            line("Calinski-Harabasz", row.get("Calinski_Harabasz")),
            line("Davies-Bouldin", row.get("Davies_Bouldin")),
        ])
        return (
            f"<div style='font-size:15px; font-weight:bold; padding-bottom:6px;'>"
            f"{row.get('model', '')} &nbsp;·&nbsp; {self._pipeline_str(row)}</div>"
            f"<div style='opacity:0.7; padding-bottom:10px;'>{self._fmt_ts(row.get('timestamp', ''))}</div>"
            f"<table cellspacing='0'>"
            + line("Dataset", row.get("dataset"))
            + line("Images", row.get("n_images"))
            + line("Clusters", f"{row.get('n_clusters', '?')} ({row.get('noise_pct', '0')}% noise)")
            + line("Device", row.get("device"))
            + line("Labeled", row.get("labeled"))
            + line("Reduction", f"{row.get('reduction', '')} ({params(row.get('reduction_params'))})")
            + line("Clustering", f"{row.get('clustering', '')} ({params(row.get('clustering_params'))})")
            + line("Seed", row.get("seed"))
            + "</table>"
            + "<div style='font-weight:bold; padding:12px 0 4px 0;'>Metrics</div>"
            + f"<table cellspacing='0'>{metrics_rows}</table>")

    def _best_history_row(self, rows):
        """Best = highest ARI among labeled runs, else highest Silhouette."""
        ari = [(r, self._fnum(r.get("ARI"))) for r in rows if self._fnum(r.get("ARI")) is not None]
        if ari:
            return max(ari, key=lambda t: t[1])[0]
        sil = [(r, self._fnum(r.get("Silhouette"))) for r in rows if self._fnum(r.get("Silhouette")) is not None]
        return max(sil, key=lambda t: t[1])[0] if sil else None

    def _history_selected_row(self):
        r = self.history_table.currentRow()
        if r < 0:
            return None
        item0 = self.history_table.item(r, 0)
        return item0.data(Qt.UserRole) if item0 else None

    def _history_apply_selected(self):
        obj = self._history_selected_row()
        if not obj:
            QMessageBox.information(self, "No selection",
                                    "Select a row in the table first.")
            return
        # For a grouped config, apply its best-scoring run's settings.
        row = obj["rep"] if isinstance(obj, dict) and obj.get("_group") else obj
        self._apply_settings_from_row(row)

    def _apply_settings_from_row(self, row):
        """Re-populate the sidebar from a past run so it can be reproduced."""
        import json
        try:
            # Model + device (stored as the combo's userData values)
            idx = self.combo_model.findData(row.get("model"))
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
            idx = self.combo_device.findData(row.get("device"))
            if idx >= 0:
                self.combo_device.setCurrentIndex(idx)

            # Reduction / clustering methods -> these rebuild the param widgets
            # synchronously via their currentTextChanged slots.
            if row.get("reduction"):
                self.combo_reduction.setCurrentText(row["reduction"])
            if row.get("clustering"):
                self.combo_clustering.setCurrentText(row["clustering"])

            # Seed
            seed = self._fnum(row.get("seed"))
            if seed is not None:
                self.spin_seed.setValue(int(seed))

            # Params (written back into the freshly-rebuilt widgets)
            rp = json.loads(row.get("reduction_params") or "{}")
            cp = json.loads(row.get("clustering_params") or "{}")
            self._set_param_widgets(self.reduction_widgets, rp)
            self._set_param_widgets(self.clustering_widgets, cp)

            self.lbl_status.setText(
                f"Applied settings from run: {row.get('model')} + "
                f"{row.get('reduction')} + {row.get('clustering')}. Press Run to reproduce.")
        except Exception as e:
            QMessageBox.warning(self, "Apply failed", str(e))

    def _set_param_widgets(self, store, params):
        """Write stored param values into the current dynamic widgets, skipping
        any that no longer exist (e.g. a removed hyperparameter)."""
        for key, val in params.items():
            w = store.get(key)
            if w is None:
                continue
            try:
                if isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(float(val)))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(val))
            except Exception:
                pass

    def _history_open_folder(self):
        try:
            os.makedirs(_HISTORY_DIR, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(_HISTORY_DIR)   # Windows
            else:
                QMessageBox.information(self, "Location", _HISTORY_DIR)
        except Exception as e:
            QMessageBox.warning(self, "Open failed", str(e))

    def _history_delete_selected(self):
        import csv
        obj = self._history_selected_row()
        if not obj:
            QMessageBox.information(self, "No selection", "Select a row first.")
            return
        if isinstance(obj, dict) and obj.get("_group"):
            to_delete = obj["runs"]
            msg = f"Delete all {len(to_delete)} run(s) of this configuration?"
        else:
            to_delete = [obj]
            msg = "Remove this run from the history?"
        if QMessageBox.question(self, "Delete", msg) != QMessageBox.Yes:
            return
        del_uids = {x.get("_uid") for x in to_delete if x.get("_uid") is not None}
        if not del_uids:
            QMessageBox.warning(self, "Delete failed",
                                "Could not identify the selected run(s). "
                                "Refresh the history and try again.")
            return
        remaining = [x for x in self._history_rows if x.get("_uid") not in del_uids]
        try:
            with open(_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_HISTORY_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for x in remaining:
                    w.writerow({c: x.get(c, "") for c in _HISTORY_COLUMNS})
        except Exception as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        self._load_history_table()

    def pipeline_finished(self, result):
        self._reset_run_button()
        # Cache the model that was just built/used so the next run can reuse
        # it (same model+device) instead of reloading onto the GPU.
        w = getattr(self, "worker", None)
        if w is not None and getattr(w, "extractor", None) is not None:
            self._cached_extractor = w.extractor
            self._cached_extractor_key = w.model_key
        self.lbl_status.setText("Generating Dashboard...")
        self.generate_dashboard(result)
        n_images   = len(result["image_paths"])
        n_clusters = len(set(result["cluster_ids"])) - (1 if -1 in result["cluster_ids"] else 0)
        self._log_run_history(result, n_images, n_clusters)
        self._load_history_table()
        self.lbl_status.setText(f"Done — {n_images:,} images → {n_clusters} clusters")
        self.btn_export.setVisible(True)
        # A new clustering result invalidates any previous annotation overlay.
        self.annotation_df = None
        self._ann_lookup = {}
        self.lbl_annot_status.setText("")
        # Cache the loaded model so annotation reuses it (skips a 2nd load).
        # None when features came from cache (model wasn't loaded this run).
        ext = result.get("extractor")
        if ext is not None:
            self._cached_extractor = ext
            self._cached_extractor_model = result.get("extractor_model")
        model_key = self.combo_model.currentData()
        self.btn_annotate.setEnabled(model_has_text(model_key))

    def pipeline_cancelled(self):
        self._reset_run_button()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Cancelled.")
        self.btn_export.setVisible(False)

    def pipeline_error(self, err_msg, tb):
        self._reset_run_button()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Error occurred.")
        self.btn_export.setVisible(False)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Pipeline Error")
        msg.setText(err_msg)
        msg.setDetailedText(tb)
        msg.exec()

    def export_results(self):
        if not hasattr(self, 'current_result'):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "cluster_results.csv", "CSV Files (*.csv)")
        if not path:
            return
        result = self.current_result
        ann = self._ann_lookup if self.annotation_df is not None else {}
        rows = []
        for i, (img_path, cid) in enumerate(zip(result["image_paths"], result["cluster_ids"])):
            row = {
                "path":          img_path,
                "filename":      os.path.basename(img_path),
                "cluster_id":    cid,
                "cluster_label": "Noise" if cid == -1 else str(cid),
            }
            if result.get("true_labels") is not None:
                row["true_class"] = result["class_names"][result["true_labels"][i]]
            r = ann.get(str(img_path))
            if r is not None:
                row["ai_annotation"]        = r.get("ai_annotation")
                row["ai_second_choice"]     = r.get("ai_second_choice")
                row["margin"]               = r.get("margin")
                row["combined_uncertainty"] = r.get("combined_uncertainty")
                row["confidence_tier"]      = r.get("confidence_tier")
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Export Complete", f"Results saved to:\n{path}")

    def _show_figure(self, view, fig, name):
        """Render a Plotly figure into a QWebEngineView via a local file.

        setHtml() cannot be used here: Qt caps it at 2 MB and an embedded
        plotly.js pushes a figure to ~4.9 MB, which silently renders a blank
        view. Loading from a file URL has no such limit. plotly.js is written
        once per session next to the figures and referenced by name, so the
        charts stay fully offline -- a CDN reference renders blank in the
        packaged build when there is no internet.
        """
        d = self._plot_dir()
        js = os.path.join(d, "plotly.min.js")
        if not os.path.exists(js):
            from plotly.offline import get_plotlyjs
            with open(js, "w", encoding="utf-8") as f:
                f.write(get_plotlyjs())

        path = os.path.join(d, f"{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(fig.to_html(include_plotlyjs="directory", full_html=True))
        view.setUrl(QUrl.fromLocalFile(path))

    def _plot_dir(self):
        """Session-scoped scratch directory for rendered figures."""
        if getattr(self, "_plot_tmp", None) is None:
            import tempfile
            self._plot_tmp = tempfile.mkdtemp(prefix="biocluster_plots_")
        return self._plot_tmp

    def show_credits(self, *_):
        _CreditsDialog(self).exec()

    def closeEvent(self, event):
        """Drop the rendered-figure scratch directory on exit -- it holds a
        ~4.8 MB copy of plotly.js and would otherwise accumulate per session."""
        tmp = getattr(self, "_plot_tmp", None)
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        super().closeEvent(event)

    def _create_metric_card(self, title, value, metric=None, tooltip="",
                            raw=None):
        """Metric card. `metric` is the canonical metric key -- the accent stripe
        is derived from the value via T.status_for, so the colour MEANS
        good/warning/poor instead of being decoration. Metrics with no absolute
        'good' value (Calinski-Harabasz, Davies-Bouldin) get a neutral card."""
        card = QFrame()
        # Colour from the NUMERIC value. `value` may be display-formatted --
        # Hungarian renders as "72.74%", which float() cannot parse, so the
        # card silently lost its status stripe. `raw` carries the number.
        accent = T.status_for(metric, value if raw is None else raw) if metric else None
        card.setStyleSheet(T.card_qss(accent))
        layout = QVBoxLayout(card)
        layout.setContentsMargins(T.SPACE_LG, T.SPACE_LG, T.SPACE_LG, T.SPACE_LG)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            f"font-size: {T.TYPE_CAPTION}px; color: {T.INK_MUTED}; "
            f"font-weight: 600; border: none;")

        lbl_value = QLabel(str(value))
        lbl_value.setStyleSheet(f"color: {T.INK}; border: none;")
        font2 = QFont()
        font2.setPointSize(T.TYPE_HERO)
        font2.setBold(True)
        lbl_value.setFont(font2)
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        layout.addStretch()
        if tooltip:
            card.setToolTip(tooltip)
        return card

    def generate_dashboard(self, result):
        self.current_result = result
        image_paths = result["image_paths"]
        viz_features = result["viz_features"]
        cluster_ids = result["cluster_ids"]
        metrics = result["metrics"]
        class_names = result.get("class_names")

        names = [os.path.basename(p) for p in image_paths]
        
        df = pd.DataFrame({
            'Filename': names,
            'Path': image_paths,
            'Cluster': [str(c) if c != -1 else 'Noise' for c in cluster_ids]
        })
        
        # Add True Class for hover if available
        if result.get("true_labels") is not None:
            df['True Class'] = [class_names[t] for t in result["true_labels"]]
            hover_data = {'Filename': True, 'True Class': True, 'x': False, 'y': False}
        else:
            hover_data = {'Filename': True, 'x': False, 'y': False}

        # --- Tab 1: 2D Projection Map ---
        df['x'] = viz_features[:, 0]
        df['y'] = viz_features[:, 1]
        
        # Below the categorical limit, colour is a true identity key: fixed
        # palette + legend. Above it, no palette can name 200 clusters, but
        # colour still does real work as a GROUPING cue -- it shows which
        # points travel together even when you cannot name the group. So we
        # widen to a large qualitative set and drop the legend, which at 200
        # rows was noise; hover carries the exact cluster ID.
        n_groups   = df['Cluster'].nunique()
        many       = n_groups > T.CATEGORICAL_MAX
        color_seq  = (px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
                      if many else T.CATEGORICAL)
        
        fig_proj = px.scatter(df, x='x', y='y', color='Cluster',
                              hover_name='Filename', 
                              hover_data=hover_data,
                              template="plotly_white",
                              color_discrete_sequence=color_seq,
                              title="2D Projection of Clustered Features")
        
        fig_proj.update_traces(marker=dict(size=7, opacity=0.85,
                                           line=dict(width=2, color=T.SURFACE)))
        fig_proj.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Assigned Cluster",
            showlegend=not many,          # a 200-row legend is noise, not information
            font=dict(family=T.FONT_FAMILY, size=T.TYPE_SUBTITLE),
            paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE,
        )
        fig_proj.update_xaxes(gridcolor=T.GRID, zeroline=False)
        fig_proj.update_yaxes(gridcolor=T.GRID, zeroline=False)
        if many:
            fig_proj.update_layout(title=(
                f"2D Projection of Clustered Features"
                f"<br><sup>{n_groups} clusters — hover a point for its cluster ID</sup>"))
        self._show_figure(self.plot_view, fig_proj, "projection")

        # --- Tab 2: Cluster Distribution ---
        counts = df['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', 'Count']
        counts['sort_key'] = counts['Cluster'].apply(lambda x: -1 if x == 'Noise' else int(x))
        counts = counts.sort_values('sort_key').drop('sort_key', axis=1)

        # Colour each bar by cluster using the SAME sequence as the projection
        # map. On a lone chart one colour would be the cleaner choice -- bar
        # length already encodes the value -- but here colour is the key that
        # links the two views, so a cluster can be found in one from the other.
        # Consistency across the pair beats purity within one chart.
        fig_dist = px.bar(counts, x='Cluster', y='Count', color='Cluster',
                          template="plotly_white",
                          color_discrete_sequence=color_seq,
                          title="Image Count per Cluster")
        fig_dist.update_layout(showlegend=not many)
        # A border drawn around marks to separate them is an anti-pattern; a
        # 2px surface-coloured gap does the same job without the visual noise.
        fig_dist.update_traces(marker_line_width=2, marker_line_color=T.SURFACE)
        fig_dist.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_title="Cluster ID",
            yaxis_title="Number of Images",
            showlegend=False,
            font=dict(family="sans-serif", size=13),
        )
        self._show_figure(self.dist_view, fig_dist, "distribution")
        
        # --- Tab 3: Evaluation Metrics ---
        self._clear_layout(self.cards_layout)
        
        if metrics['type'] == 'external':
            self.metrics_title.setText("External Clustering Metrics (Ground Truth)")
            self.metrics_desc.setText("These metrics evaluate how perfectly the unsupervised clusters match the actual biological taxonomy.")
            
            ari_val = f"{metrics['ARI']:.4f}" if isinstance(metrics['ARI'], float) else metrics['ARI']
            nmi_val = f"{metrics['NMI']:.4f}" if isinstance(metrics['NMI'], float) else metrics['NMI']
            acc_val = f"{metrics['Hungarian Accuracy']:.2%}" if isinstance(metrics['Hungarian Accuracy'], float) else metrics['Hungarian Accuracy']
            
            self.cards_layout.addWidget(self._create_metric_card("Adjusted Rand Index (ARI)", ari_val, "ARI",
                tooltip="Range: -1 to 1. Higher is better. 1.0 = perfect match with ground truth. 0 = random."))
            self.cards_layout.addWidget(self._create_metric_card("Normalized Mutual Info (NMI)", nmi_val, "NMI",
                tooltip="Range: 0 to 1. Higher is better. Measures shared information between predicted clusters and true labels."))
            self.cards_layout.addWidget(self._create_metric_card("Hungarian Accuracy", acc_val, "Hungarian",
                raw=metrics.get("Hungarian Accuracy"),
                tooltip="Range: 0% to 100%. Higher is better. Optimal one-to-one assignment of clusters to true classes."))
        else:
            self.metrics_title.setText("Internal Clustering Metrics (No Ground Truth)")
            self.metrics_desc.setText("Because no labels were provided, these metrics evaluate the mathematical cohesion and separation of the clusters.")
            
            sil_val = f"{metrics['Silhouette Score']:.4f}" if isinstance(metrics['Silhouette Score'], float) else metrics['Silhouette Score']
            ch_val = f"{metrics['Calinski-Harabasz']:.1f}" if isinstance(metrics['Calinski-Harabasz'], float) else metrics['Calinski-Harabasz']
            db_val = f"{metrics['Davies-Bouldin']:.4f}" if isinstance(metrics['Davies-Bouldin'], float) else metrics['Davies-Bouldin']
            
            self.cards_layout.addWidget(self._create_metric_card("Silhouette Score", sil_val, "Silhouette",
                tooltip="Range: -1 to 1. Higher is better. Measures how tightly packed clusters are vs. how well-separated they are."))
            self.cards_layout.addWidget(self._create_metric_card("Calinski-Harabasz Index", ch_val, None,
                tooltip="No fixed range. Higher is better. Ratio of between-cluster to within-cluster dispersion."))
            self.cards_layout.addWidget(self._create_metric_card("Davies-Bouldin Index", db_val, None,
                tooltip="Range: 0 to ∞. Lower is better. Average similarity between each cluster and its most similar neighbour."))

        # --- Tab 4: Populate Gallery Dropdown ---
        cluster_names = list(df['Cluster'].unique())
        if 'Noise' in cluster_names:
            cluster_names.remove('Noise')
            cluster_names = sorted(cluster_names, key=int)
            cluster_names.insert(0, 'Noise')
        else:
            cluster_names = sorted(cluster_names, key=int)
        self._present_clusters = cluster_names
        self._tier_filter = "all"
        self._cluster_tier = {}          # tiers only known after annotation
        # A fresh clustering invalidates any prior corrections.
        self.decisions = {}
        self._undo, self._redo = [], []
        self._autosave_path = None
        self._img_correct = {}
        self._overall_acc = None
        self._cluster_acc = None
        self._overall_acc_n = (0, 0)
        self._cluster_acc_n = (0, 0)
        self._set_correction_enabled(False)
        for _k, _b in getattr(self, "_filter_buttons", {}).items():
            _b.setChecked(_k == "all")
        self._populate_cluster_dropdown()

        # Switch to the first tab upon completion
        self.tabs.setCurrentIndex(0)

    def _populate_cluster_dropdown(self):
        """Fill the cluster dropdown, applying the tier filter and (once
        annotated) worst-first ordering (human_required -> review -> auto)."""
        names = list(self._present_clusters)
        f = self._tier_filter
        if f == "noise":
            names = [n for n in names if n == "Noise"]
        elif f != "all" and self._cluster_tier:
            names = [n for n in names if n != "Noise" and self._cluster_tier.get(n) == f]
        if self._cluster_tier:
            prio = {"human_required": 0, "review": 1, "auto_accept": 2}
            names = sorted(names, key=lambda n: (9, 0) if n == "Noise"
                           else (prio.get(self._cluster_tier.get(n), 3), int(n)))
        self.combo_cluster_view.blockSignals(True)
        self.combo_cluster_view.clear()
        self.combo_cluster_view.addItems(names)
        self.combo_cluster_view.blockSignals(False)
        if names:
            self.update_gallery(names[0])
        else:
            self._clear_layout(self.gallery_grid)
            self.lbl_gallery_header.setText("No clusters match this filter.")
            self.lbl_cluster_signals.setText("")

    def _set_tier_filter(self, key):
        self._tier_filter = key
        for k, b in self._filter_buttons.items():
            b.setChecked(k == key)
        self._populate_cluster_dropdown()

    def update_gallery(self, cluster_name):
        self._gallery_shown = 60
        self._render_gallery(cluster_name)

    def _render_gallery(self, cluster_name):
        if not cluster_name or not hasattr(self, 'current_result'):
            return

        self._clear_layout(self.gallery_grid)

        image_paths = self.current_result["image_paths"]
        cluster_ids = self.current_result["cluster_ids"]
        true_labels = self.current_result.get("true_labels")
        class_names = self.current_result.get("class_names")

        target_id = -1 if cluster_name == 'Noise' else int(cluster_name)

        indices = [i for i, cid in enumerate(cluster_ids) if cid == target_id]
        header = f"Cluster {cluster_name}  ·  {len(indices)} images"

        # If annotated, append the cluster's AI label / runner-up / tier, and
        # build the cluster-level signal summary line (like the web view).
        cluster_sig_text = ""
        ref_top1, ref_runner = None, None
        if self.annotation_df is not None and target_id != -1 and indices:
            rows = [self._ann_lookup.get(str(image_paths[i])) for i in indices]
            rows = [r for r in rows if r is not None]
            if rows:
                label = str(rows[0]["ai_annotation"])
                runner = str(rows[0].get("ai_second_choice", "") or "")
                ref_top1, ref_runner = label, (runner or None)
                margin = float(rows[0].get("margin", 0.0) or 0.0)
                order = {"human_required": 0, "review": 1, "auto_accept": 2}
                worst = min((str(r.get("confidence_tier", "auto_accept")) for r in rows),
                            key=lambda t: order.get(t, 3))
                header += (f"   —   AI: {label}"
                           + (f"  ·  runner-up: {runner} (Δ{margin:.3f})" if runner else "")
                           + f"  ·  {worst}")
                # Per-cluster annotation accuracy (when ground truth exists).
                caccs = [self._img_correct.get(str(image_paths[i])) for i in indices]
                caccs = [a for a in caccs if a is not None]
                if caccs:
                    header += f"  ·  acc {100 * sum(caccs) / len(caccs):.0f}%"

                def _mean(field):
                    vals = [float(r.get(field, 0.0) or 0.0) for r in rows]
                    return sum(vals) / len(vals) if vals else 0.0
                sem, spa = _mean("semantic_uncertainty"), _mean("spatial_uncertainty")
                agr, tax = _mean("cluster_agreement"), _mean("taxonomic_consistency")
                # Cohesion-based cluster confidence, identical to the web view.
                cluster_conf = max(0.0, min(1.0, 1.0 - (spa * 0.5 + agr * 0.5)))
                cluster_sig_text = (
                    f"Signals (cluster mean):   S {sem:.2f}    P {spa:.2f}    "
                    f"A {agr:.2f}    T {tax:.2f}        ·        "
                    f"Cluster conf {cluster_conf * 100:.0f}%")
        self.lbl_gallery_header.setText(header)
        self.lbl_cluster_signals.setText(cluster_sig_text)
        self._render_references(ref_top1, ref_runner)
        indices_to_show = indices[:self._gallery_shown]

        row, col = 0, 0
        cols_per_row = 5
        for idx in indices_to_show:
            path = image_paths[idx]
            
            container = QWidget()
            cont_layout = QVBoxLayout(container)
            cont_layout.setContentsMargins(5,5,5,5)
            cont_layout.setSpacing(5)
            
            lbl = _ClickableLabel(str(path))
            lbl.clicked.connect(self._zoom_image)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pixmap)
            else:
                lbl.setText("Corrupt Image")

            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(180, 180)
            
            # Border colour = confidence tier when annotated; corrected images
            # (noise/junk) override with their own colour.
            TIER_COLOR = {"auto_accept": "#2ed573", "review": "#ffa502",
                          "human_required": "#ff4757"}
            STATUS_COLOR = {"noise": "#e67e22", "junk": "#c0392b"}
            row_ann = (self._ann_lookup.get(str(path))
                       if self.annotation_df is not None else None)
            dec = self.decisions.get(str(path))
            border_col = "#d1d8e0"
            if row_ann is not None:
                border_col = TIER_COLOR.get(str(row_ann.get("confidence_tier", "")), "#d1d8e0")
            if dec is not None and dec.get("status") in STATUS_COLOR:
                border_col = STATUS_COLOR[dec["status"]]
            container.setStyleSheet(f"border: 3px solid {border_col}; border-radius: 4px;")

            cont_layout.addWidget(lbl)

            if true_labels is not None:
                true_class = class_names[true_labels[idx]]
                lbl_true = QLabel(f"True: {true_class}")
                lbl_true.setAlignment(Qt.AlignCenter)
                lbl_true.setStyleSheet("border: none; font-weight: bold; color: #1e90ff;")
                cont_layout.addWidget(lbl_true)
                # Per-image annotation correctness (AI label vs true class).
                ok = self._img_correct.get(str(path))
                if ok is not None:
                    lbl_ok = QLabel("✓ correct" if ok else "✗ wrong")
                    lbl_ok.setAlignment(Qt.AlignCenter)
                    lbl_ok.setStyleSheet(
                        "border:none; font-size:10px; font-weight:bold; color:"
                        + ("#16a34a" if ok else "#dc2626") + ";")
                    cont_layout.addWidget(lbl_ok)
            else:
                fname = os.path.basename(image_paths[idx])
                if len(fname) > 22:
                    fname = fname[:19] + "..."
                lbl_fname = QLabel(fname)
                lbl_fname.setAlignment(Qt.AlignCenter)
                lbl_fname.setStyleSheet("border: none; font-size: 10px; color: #57606f;")
                lbl_fname.setToolTip(image_paths[idx])
                cont_layout.addWidget(lbl_fname)

            # Annotation overlay: per-image confidence, tier, and S/P/A/T.
            if row_ann is not None:
                cu = float(row_ann.get("combined_uncertainty", 0.0) or 0.0)
                conf = max(0.0, min(1.0, 1.0 - cu))
                tier = str(row_ann.get("confidence_tier", ""))
                tier_col = TIER_COLOR.get(tier, "#57606f")
                lbl_conf = QLabel(
                    f"conf {conf * 100:.0f}%  ·  unc {cu:.2f}  ·  {tier.replace('_', ' ')}")
                lbl_conf.setAlignment(Qt.AlignCenter)
                lbl_conf.setStyleSheet(
                    f"border: none; font-size: 10px; font-weight: bold; color: {tier_col};")
                lbl_conf.setToolTip(
                    "conf = 1 - combined uncertainty (unc).\n"
                    "Tier is decided from unc vs thresholds:\n"
                    f"  unc < {self.spin_rev.value():.2f}  -> auto_accept\n"
                    f"  {self.spin_rev.value():.2f} <= unc < {self.spin_hum.value():.2f}  -> review\n"
                    f"  unc >= {self.spin_hum.value():.2f}  -> human_required")
                cont_layout.addWidget(lbl_conf)

                s = float(row_ann.get("semantic_uncertainty", 0.0) or 0.0)
                p = float(row_ann.get("spatial_uncertainty", 0.0) or 0.0)
                a = float(row_ann.get("cluster_agreement", 0.0) or 0.0)
                t = float(row_ann.get("taxonomic_consistency", 0.0) or 0.0)
                lbl_sig = QLabel(f"S{s:.2f}  P{p:.2f}  A{a:.2f}  T{t:.2f}")
                lbl_sig.setAlignment(Qt.AlignCenter)
                lbl_sig.setStyleSheet("border: none; font-size: 9px; color: #8395a7;")
                lbl_sig.setToolTip("Per-image uncertainty (0-1, higher = worse): "
                                   "S semantic, P spatial, A agreement, T taxonomic")
                cont_layout.addWidget(lbl_sig)

                # --- Per-image correction controls ---
                cur = (dec or {}).get("final", str(row_ann.get("ai_annotation", "")))
                status = (dec or {}).get("status", "species")

                # Quick set: top-1 / runner-up
                ai1 = str(row_ann.get("ai_annotation", "") or "")
                ai2 = str(row_ann.get("ai_second_choice", "") or "")
                quick = QHBoxLayout(); quick.setSpacing(3)
                b_top1 = QPushButton("✓ top-1")
                b_top1.setToolTip(ai1)
                b_top1.clicked.connect(partial(self._set_image_label, str(path), ai1))
                b_top1.setStyleSheet("border: none; font-size: 10px; padding: 2px;")
                quick.addWidget(b_top1)
                if ai2:
                    b_run = QPushButton("→ runner-up")
                    b_run.setToolTip(ai2)
                    b_run.clicked.connect(partial(self._set_image_label, str(path), ai2))
                    b_run.setStyleSheet("border: none; font-size: 10px; padding: 2px;")
                    quick.addWidget(b_run)
                qw = QWidget(); qw.setLayout(quick); qw.setStyleSheet("border: none;")
                cont_layout.addWidget(qw)

                relabel = _ComboBox()
                relabel.setEditable(True)
                if self._candidates:
                    relabel.addItems(self._candidates)
                relabel.setCurrentText(cur)
                relabel.setStyleSheet("border: 1px solid #d1d8e0; font-size: 10px;")
                relabel.setToolTip("Relabel this image (type or pick a species)")
                relabel.currentTextChanged.connect(partial(self._set_image_label, str(path)))
                cont_layout.addWidget(relabel)

                actions = QHBoxLayout(); actions.setSpacing(3)
                b_noise = QPushButton("noise"); b_junk = QPushButton("junk"); b_reset = QPushButton("reset")
                b_noise.clicked.connect(partial(self._set_image_status, str(path), "noise"))
                b_junk.clicked.connect(partial(self._set_image_status, str(path), "junk"))
                b_reset.clicked.connect(partial(self._reset_image, str(path)))
                for b in (b_noise, b_junk, b_reset):
                    b.setStyleSheet("border: none; font-size: 10px; padding: 2px;")
                    actions.addWidget(b)
                aw = QWidget(); aw.setLayout(actions); aw.setStyleSheet("border: none;")
                cont_layout.addWidget(aw)

                if status in ("noise", "junk"):
                    scol = STATUS_COLOR[status]
                    lbl_status = QLabel(status.upper())
                    lbl_status.setAlignment(Qt.AlignCenter)
                    lbl_status.setStyleSheet(
                        f"border: none; font-size: 10px; font-weight: bold; color: {scol};")
                    cont_layout.addWidget(lbl_status)

            self.gallery_grid.addWidget(container, row, col)

            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

        if len(indices) > self._gallery_shown:
            remaining = len(indices) - self._gallery_shown
            btn_more = QPushButton(f"Show 60 More  ({remaining} remaining)")
            _set_role(btn_more, "accent")
            btn_more.clicked.connect(self._show_more_gallery)
            next_row = row + (1 if col > 0 else 0)
            self.gallery_grid.addWidget(btn_more, next_row, 0, 1, cols_per_row)

    def _show_more_gallery(self):
        self._gallery_shown += 60
        self._render_gallery(self.combo_cluster_view.currentText())

    def _zoom_image(self, path):
        """Open a large modal view of a thumbnail (click to enlarge)."""
        pix = QPixmap(path)
        if pix.isNull():
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        dlg.setStyleSheet("background:#111;")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel()
        geo = self.screen().availableGeometry()
        lbl.setPixmap(pix.scaled(int(geo.width() * 0.9), int(geo.height() * 0.9),
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        dlg.exec()

    def _render_references(self, top1, runner):
        """Show up to 3 dataset exemplars for the cluster's top-1 / runner-up."""
        self._clear_layout(self.ref_layout)
        if not self._ref_map:
            return
        for lab, tag, col in ((top1, "Ref (top-1)", "#1e90ff"),
                              (runner, "Ref (runner-up)", "#7c3aed")):
            paths = self._ref_map.get(lab, []) if lab else []
            if not paths:
                continue
            box = QVBoxLayout()
            t = QLabel(f"{tag}: {lab}")
            t.setStyleSheet(f"border:none; font-size:11px; font-weight:bold; color:{col};")
            box.addWidget(t)
            rowl = QHBoxLayout(); rowl.setSpacing(4)
            for p in paths[:3]:
                pm = QPixmap(p)
                if pm.isNull():
                    continue
                rl = _ClickableLabel(str(p))
                rl.setPixmap(pm.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                rl.setFixedSize(90, 90)
                rl.setStyleSheet("border:1px solid #d1d8e0;")
                rl.clicked.connect(self._zoom_image)
                rowl.addWidget(rl)
            box.addLayout(rowl)
            cw = QWidget(); cw.setLayout(box)
            self.ref_layout.addWidget(cw)

    # -------------------- Corrections --------------------

    def _set_correction_enabled(self, on):
        for w in (self.btn_confirm_cluster, self.btn_confirm_runnerup, self.combo_relabel,
                  self.btn_apply_relabel, self.btn_cluster_noise, self.btn_cluster_junk,
                  self.btn_export_decisions, self.btn_save_session, self.btn_load_session,
                  self.btn_web_report):
            w.setEnabled(on)
        for b in getattr(self, "_filter_buttons", {}).values():
            b.setEnabled(on)
        if on:
            self._update_undo_redo()
        else:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)

    # ---- undo / redo / autosave / session ----

    def _snapshot(self):
        """Record the current decisions before a change (for undo)."""
        self._undo.append(json.dumps(self.decisions))
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._redo.clear()

    def _after_change(self, rerender=True):
        self._autosave()
        self._update_undo_redo()
        if rerender and self.combo_cluster_view.count():
            self._render_gallery(self.combo_cluster_view.currentText())

    def _update_undo_redo(self):
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_redo.setEnabled(bool(self._redo))

    def _autosave(self):
        if not self._autosave_path:
            return
        try:
            os.makedirs(os.path.dirname(self._autosave_path), exist_ok=True)
            with open(self._autosave_path, "w", encoding="utf-8") as f:
                json.dump(self.decisions, f)
        except Exception:
            pass

    def undo_decision(self):
        if not self._undo:
            return
        self._redo.append(json.dumps(self.decisions))
        self.decisions = json.loads(self._undo.pop())
        self._after_change()

    def redo_decision(self):
        if not self._redo:
            return
        self._undo.append(json.dumps(self.decisions))
        self.decisions = json.loads(self._redo.pop())
        self._after_change()

    def save_session(self):
        if not self.decisions:
            QMessageBox.information(self, "Nothing to save", "Run annotation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "review_session.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"decisions": self.decisions}, f, indent=2)
        QMessageBox.information(self, "Saved", f"Session saved to:\n{path}")

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            dec = data.get("decisions", data) if isinstance(data, dict) else None
            if not isinstance(dec, dict):
                raise ValueError("not a session file")
            self._snapshot()
            self.decisions = {str(k): v for k, v in dec.items()}
            self._after_change()
            QMessageBox.information(self, "Loaded",
                f"Loaded {len(self.decisions)} decisions from session.")
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    # ---- corrections ----

    def _set_image_label(self, path, text):
        text = (text or "").strip()
        if not text:
            return
        self._snapshot()
        self.decisions[path] = {"final": text, "status": "species"}
        self._after_change(rerender=False)   # the combo already shows the text

    def _set_image_status(self, path, status):
        self._snapshot()
        d = self.decisions.get(path) or {"final": ""}
        d["status"] = status
        self.decisions[path] = d
        self._after_change()

    def _reset_image(self, path):
        r = self._ann_lookup.get(path)
        if r is not None:
            self._snapshot()
            self.decisions[path] = {"final": str(r["ai_annotation"]), "status": "species"}
        self._after_change()

    def _current_cluster_paths(self):
        if not hasattr(self, "current_result"):
            return []
        name = self.combo_cluster_view.currentText()
        if not name:
            return []
        target = -1 if name == "Noise" else int(name)
        cids = self.current_result["cluster_ids"]
        paths = self.current_result["image_paths"]
        return [str(paths[i]) for i, c in enumerate(cids) if c == target]

    def confirm_cluster(self):
        self._snapshot()
        for p in self._current_cluster_paths():
            r = self._ann_lookup.get(p)
            if r is not None:
                self.decisions[p] = {"final": str(r["ai_annotation"]), "status": "species"}
        self._after_change()

    def confirm_cluster_runnerup(self):
        paths = self._current_cluster_paths()
        if not paths:
            return
        r = self._ann_lookup.get(paths[0])
        runner = str(r.get("ai_second_choice", "") or "") if r is not None else ""
        if not runner:
            QMessageBox.information(self, "No runner-up",
                                    "This cluster has no runner-up species.")
            return
        self._snapshot()
        for p in paths:
            self.decisions[p] = {"final": runner, "status": "species"}
        self._after_change()

    def relabel_cluster(self):
        label = self.combo_relabel.currentText().strip()
        if not label:
            QMessageBox.information(self, "No label", "Type or pick a species to apply.")
            return
        self._snapshot()
        for p in self._current_cluster_paths():
            self.decisions[p] = {"final": label, "status": "species"}
        self._after_change()

    def mark_cluster(self, status):
        self._snapshot()
        for p in self._current_cluster_paths():
            d = self.decisions.get(p) or {"final": ""}
            d["status"] = status
            self.decisions[p] = d
        self._after_change()

    def export_decisions(self):
        if not self.decisions:
            QMessageBox.information(self, "Nothing to export", "Run annotation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Decisions", "decisions.csv", "CSV Files (*.csv)")
        if not path:
            return
        rows = [{"file_path": p,
                 "final_label": d.get("final", ""),
                 "status": d.get("status", "species")}
                for p, d in self.decisions.items()]
        pd.DataFrame(rows, columns=["file_path", "final_label", "status"]).to_csv(
            path, index=False)
        QMessageBox.information(self, "Exported",
            f"{len(rows)} decisions saved to:\n{path}\n\n"
            "Compatible with the pipeline's apply_decisions().")

    def _fetch_gbif_reference_map(self):
        """
        Fetch GBIF photographs for the species the report will display.

        Returns the map, or None if there was nothing to look up or the user
        cancelled. Only the names shown in a reference panel are fetched, so
        this is bounded by the cluster count rather than the candidate list.

        The lookup is two HTTP requests per uncached species, so it runs
        behind a cancellable progress dialog. processEvents keeps the dialog
        responsive from inside the collector's progress callback, which is
        cheaper here than moving a secondary feature onto a worker thread.
        """
        from biocluster.annotation.gbif_media import collect_reference_urls
        from biocluster.annotation.interactive_report import displayed_species

        names = displayed_species(self.annotation_df)
        if not names:
            QMessageBox.information(self, "No species to look up",
                "Annotation produced no species names, so there is nothing "
                "to fetch reference photographs for.")
            return None

        cached = os.path.join(os.path.expanduser("~"), "BioCluster", "gbif_cache")
        dlg = QProgressDialog(
            f"Fetching GBIF reference photographs for {len(names)} species...",
            "Cancel", 0, len(names), self)
        dlg.setWindowTitle("GBIF reference photos")
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        def on_progress(done, total, name):
            dlg.setValue(done)
            dlg.setLabelText(f"{done}/{total}   {name}")
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            ref_map = collect_reference_urls(
                names, images_per_species=3, cache_dir=cached,
                progress=on_progress)
        except Exception as e:
            dlg.close()
            QMessageBox.warning(self, "GBIF lookup failed",
                f"Could not fetch reference photographs:\n{e}\n\n"
                "The report can still be generated without them by "
                "unticking the GBIF option.")
            return None
        dlg.close()

        found = sum(1 for v in ref_map.values() if v)
        if found == 0:
            QMessageBox.warning(self, "No photographs found",
                "GBIF returned no photographs for any of the proposed "
                "species. The report will have no reference panels.")
        return ref_map

    def generate_web_report(self):
        """Write the self-contained interactive review.html and open it."""
        if self.annotation_df is None:
            QMessageBox.information(self, "Annotate first",
                                    "Run annotation before generating the web report.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Web Report", "review.html", "HTML Files (*.html)")
        if not path:
            return
        from biocluster.annotation.interactive_report import generate_interactive_report
        source = self.combo_candidate.currentData() if hasattr(self, "combo_candidate") else "folder"
        mode = "csv" if source == "csv" else "local"

        ref_map = self._ref_map or {}
        if getattr(self, "chk_gbif_refs", None) and self.chk_gbif_refs.isChecked():
            gbif_map = self._fetch_gbif_reference_map()
            if gbif_map is None:
                return                      # cancelled, or no names to look up
            ref_map = gbif_map
            mode = "gbif"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            generate_interactive_report(self.annotation_df, ref_map, path,
                                        config=None, candidate_mode=mode)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            m = QMessageBox(self)
            m.setIcon(QMessageBox.Critical)
            m.setWindowTitle("Web report error")
            m.setText(str(e))
            m.setDetailedText(_traceback.format_exc())
            m.exec()
            return
        QApplication.restoreOverrideCursor()
        import webbrowser
        webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))
        size = os.path.getsize(path) / 1e6
        QMessageBox.information(self, "Web report generated",
            f"Saved ({size:.1f} MB) and opened in your browser:\n{path}")

def main():
    """Launch the desktop application."""
    # Many CUB/HUGO JPEGs carry slightly non-conformant embedded ICC colour
    # profiles. Qt logs "fromIccProfile: invalid tag offset alignment" once per
    # such image while building gallery thumbnails, which floods the console.
    # It is cosmetic -- Qt discards the bad profile and decodes the image
    # normally -- so silence just that one category.
    # --- TEMP diagnostic: capture Python exceptions AND native Qt crashes ---
    import faulthandler
    import traceback as _tb
    _logdir = os.path.join(os.path.expanduser("~"), "BioCluster")
    os.makedirs(_logdir, exist_ok=True)
    _errlog = os.path.join(_logdir, "startup_error.log")
    _errf = open(_errlog, "w", encoding="utf-8")
    faulthandler.enable(file=_errf)          # writes a stack even on a native crash
    _rc = 0
    try:
        from PySide6.QtCore import QLoggingCategory
        QLoggingCategory.setFilterRules("qt.gui.icc=false")

        app = QApplication(sys.argv)

        _logo = os.path.join(_LOGO_DIR, "app_logo.png")
        if os.path.exists(_logo):
            app.setWindowIcon(QIcon(_logo))

        # Splash screen
        _emblem = os.path.join(_LOGO_DIR, "Launching_emblem.png")
        splash = None
        if os.path.exists(_emblem):
            pix = QPixmap(_emblem).scaled(480, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = _SplashScreen(pix)
            splash.show()
            splash.showMessage("Initializing...")
            app.processEvents()

        qdarktheme.setup_theme("light")
        # qdarktheme installs its own application stylesheet, which replaced
        # ours wholesale -- so theme.py's rules (button roles, control heights,
        # the segmented control) never reached a single widget, and the only
        # buttons that looked styled were the ones carrying inline CSS. That is
        # what made button sizes inconsistent across the app. Append our sheet
        # so qdarktheme still supplies the base look and theme.py owns the
        # component rules on top of it.
        app.setStyleSheet(app.styleSheet() + chr(10) + T.qss())
        app.processEvents()

        window = MainWindow()

        if splash:
            splash.finish(window)

        window.show()
        _rc = app.exec()
    except BaseException:
        _tb.print_exc()                      # to console
        _tb.print_exc(file=_errf)            # to startup_error.log
        _errf.flush()
    _errf.flush()
    sys.exit(_rc)


if __name__ == "__main__":
    main()
