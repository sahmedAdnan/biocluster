"""
Embedding extraction using foundation models.

Currently supported:
  - bioclip2.5 (ViT-H/14, hf-hub:imageomics/bioclip-2.5-vith14)
  - bioclip2  (ViT-L/14, hf-hub:imageomics/bioclip-2)
  - bioclip   (ViT-B/16, hf-hub:imageomics/bioclip)
  - clip      (ViT-L/14, openai/clip-vit-large-patch14)
  - dinov2    (facebook/dinov2-large via timm or transformers)
  - dinov3    (ViT-H+/16, timm/vit_huge_plus_patch16_dinov3.lvd1689m)

All extractors follow the same interface:
    extractor = build_extractor(config)
    features  = extractor.extract(image_paths)

The returned features are a NumPy array of shape (N, D),
optionally L2-normalized based on config.model.normalize_embeddings.
"""

import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _perf_summary(tag: str, n: int, t0: float):
    """Print a one-line timing + peak-VRAM summary for an extraction run.
    Lets us compare run-to-run speed and spot VRAM saturation / throttling."""
    dt = time.time() - t0
    ips = n / dt if dt > 0 else 0.0
    peak = ""
    try:
        if torch.cuda.is_available():
            peak = f", peak VRAM {torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
    except Exception:
        pass
    print(f"[perf] {tag}: {n} imgs in {dt:.1f}s ({ips:.1f} img/s){peak}")
from typing import List

class BaseExtractor:
    """
    Abstract base class. All extractors must implement:
      - load_model()
      - extract(image_paths) -> np.ndarray
    """

    def __init__(self, config: dict):
        self.config    = config
        # Allow UI override, otherwise auto-detect
        ui_device = config["model"].get("device", "auto")
        if ui_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = ui_device

        self.normalize = config["model"].get("normalize_embeddings", True)
        self.batch_size = config["model"].get("batch_size", 32)
        self.num_workers = config["model"].get("num_workers", 0)
        self.model      = None
        self.preprocess = None

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the feature dimension (safe for zero vectors)."""
        norms = features.norm(dim=-1, keepdim=True)
        # Avoid division by zero: replace zero norms with 1.0 so the
        # zero vector stays zero rather than becoming NaN.
        norms = norms.clamp(min=1e-8)
        return features / norms

    def extract(self, image_paths: List[str]) -> np.ndarray:
        raise NotImplementedError


# -------------------------------------------------------
# BioCLIP / BioCLIP-2 (open_clip based)
# -------------------------------------------------------

class OpenCLIPExtractor(BaseExtractor):
    """
    Extractor for any model available through the open_clip library.
    This covers BioCLIP, BioCLIP-2, and standard CLIP models.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        import open_clip

        name       = self.config["model"]["name"]
        pretrained = self.config["model"]["pretrained"]
        arch       = self.config["model"].get("arch", None)

        print(f"[model] Loading {name} from '{pretrained}' ...")

        if arch is not None:
            # Standard open_clip model with separate arch + pretrained.
            # Example: arch='ViT-L-14', pretrained='openai'  (CLIP)
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(arch)
        else:
            # HF-Hub model: pretrained='hf-hub:imageomics/bioclip-2'
            self.model, _, self.preprocess = \
                open_clip.create_model_and_transforms(pretrained)
            self.tokenizer = open_clip.get_tokenizer(pretrained)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[model] Ready on device: {self.device}")

    def extract(self, image_paths: List[str], progress_fn=None, cancelled_fn=None) -> np.ndarray:
        """
        Extract image embeddings in batches.

        Args:
            image_paths:  List of full image file paths.
            progress_fn:  Optional callable(done_batches, total_batches) called after each batch.
            cancelled_fn: Optional callable() → bool; if it returns True the extraction stops
                          and this method returns None.

        Returns:
            NumPy array of shape (N, D), optionally L2-normalized, or None if cancelled.
        """
        all_features = []
        n_failed = 0
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"[extract] {len(image_paths)} images, "
              f"batch_size={self.batch_size}, {n_batches} batches")

        _t0 = time.time()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        # Determine the correct input size from the model's preprocessing
        # by running the preprocess on a dummy image once.
        dummy = Image.new("RGB", (256, 256))
        input_shape = tuple(self.preprocess(dummy).shape)  # (C, H, W)

        for batch_num, i in enumerate(tqdm(range(0, len(image_paths), self.batch_size),
                                           desc="Extracting features")):
            if cancelled_fn and cancelled_fn():
                return None

            batch_paths = image_paths[i : i + self.batch_size]

            # Load and preprocess each image in the batch
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    n_failed += 1
                    print(f"[warn] Could not load {path}: {e} - using zeros")
                    # Use a zero tensor matching the model's expected input size
                    images.append(torch.zeros(input_shape))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())
            if progress_fn:
                progress_fn(batch_num + 1, n_batches)

        if n_failed > 0:
            print(f"[warn] {n_failed}/{len(image_paths)} images failed to load")

        result = np.vstack(all_features)
        print(f"[extract] Done: {result.shape[0]} embeddings, dim={result.shape[1]}")
        _perf_summary("extract", result.shape[0], _t0)

        # Validate: no NaN values should be present in the output
        n_nan_rows = int(np.any(np.isnan(result), axis=1).sum())
        if n_nan_rows > 0:
            raise RuntimeError(
                f"NaN detected in {n_nan_rows} feature vectors after extraction. "
                f"This indicates corrupted images or model issues. "
                f"Run filter_valid_images() before extraction to remove bad files."
            )

        return result

    def encode_text(self, class_names: List[str],
                    prompt_templates: List[str] = None,
                    taxonomy_table: dict = None) -> np.ndarray:
        """
        Encode text prompts and return normalized text embeddings.
        Used by the annotation pipeline for zero-shot text matching.

        WHY ENSEMBLING:
            No single prompt perfectly represents a biological species in
            embedding space. Averaging over a small set of complementary
            prompts produces a more stable, central estimate. This is the
            same technique used in the original CLIP paper and has been
            shown to improve zero-shot alignment consistently.

        WHY TAXONOMIC TEMPLATES:
            BioCLIP-2 was trained on TreeOfLife-200M, where the text paired
            with each image is structured taxonomic text, not image captions.
            Prompts that include taxonomic context (order, family, genus) are
            therefore more aligned with the model's training distribution than
            generic "a photo of X" prompts.

            The genus is always the first word of a binomial scientific name,
            so it is parsed directly from the folder name without any external
            lookup (e.g. "abrostola_tripartita" -> genus="abrostola").

            Family and order are looked up from a taxonomy table (pre-computed
            CSV or GBIF query). If a template uses {family} or {order} but the
            taxonomy entry is missing, that template is skipped for that class.

        Args:
            class_names:       List of class folder name strings.
            prompt_templates:  List of template strings with {name}, {genus},
                               {family}, {order} placeholders.
                               If None, uses the default set below.
            taxonomy_table:    Optional dict mapping class_name -> taxonomy entry
                               with keys: genus, family, order.
                               Built by biocluster.annotation.taxonomy.

        Returns:
            NumPy array of shape (num_classes, D), L2-normalized.
            Each row is the mean of all template embeddings for that class,
            re-normalized to unit length.
        """
        # Default templates. These follow the format closest to BioCLIP-2's
        # training data (taxonomic hierarchy first), plus a few descriptive
        # alternatives to reduce variance through ensembling.
        #
        # {name}   -> full scientific name with spaces  (e.g. "abrostola tripartita")
        # {genus}  -> genus only, first word             (e.g. "abrostola")
        # {family} -> taxonomic family                   (e.g. "Noctuidae")
        # {order}  -> taxonomic order                    (e.g. "Lepidoptera")
        #
        # Override these in your YAML config under model.prompt_templates.
        if prompt_templates is None:
            prompt_templates = [
                "{name}",
                "a photo of {name}",
                "a photo of {name}, a species",
                "a species of genus {genus}",
            ]

        if taxonomy_table is None:
            taxonomy_table = {}

        print(f"[text] Encoding {len(class_names)} classes "
              f"with {len(prompt_templates)} prompt templates (ensemble) ...")
        if taxonomy_table:
            n_with_family = sum(1 for n in class_names
                                if taxonomy_table.get(n, {}).get("family"))
            print(f"[text] Taxonomy available for {n_with_family}/{len(class_names)} "
                  f"classes (family-level)")

        all_class_embeddings = []

        for name in class_names:
            # Parse the scientific name and genus from the folder name.
            # Strip numeric prefixes like "001." (CUB-200 format).
            clean = name
            if clean and clean.split(".")[0].isdigit():
                clean = clean.split(".", 1)[1]
            readable_name = clean.replace("_", " ").strip()
            genus         = readable_name.split()[0] if " " in readable_name \
                            else readable_name

            # Look up family and order from taxonomy table
            tax = taxonomy_table.get(name, {})
            family = tax.get("family", "")
            order  = tax.get("order", "")

            # Fill in the templates, skipping any that require missing fields
            filled = []
            for t in prompt_templates:
                if "{family}" in t and not family:
                    continue
                if "{order}" in t and not order:
                    continue
                try:
                    filled.append(t.format(
                        name=readable_name, genus=genus,
                        family=family, order=order,
                    ))
                except KeyError:
                    continue

            if not filled:
                # Fallback: use just the readable name
                filled = [readable_name]

            # Encode all templates for this class
            tokens   = self.tokenizer(filled).to(self.device)
            with torch.no_grad():
                embeds = self.model.encode_text(tokens)       # (n_templates, D)
                embeds = self._normalize(embeds)              # normalize each

            # Average across templates, then re-normalize.
            # Re-normalizing after averaging is important: the mean of unit
            # vectors is not itself a unit vector.
            mean_embed = embeds.mean(dim=0)                   # (D,)
            mean_embed = mean_embed / mean_embed.norm()       # back to unit sphere

            all_class_embeddings.append(mean_embed.cpu().numpy())

        result = np.stack(all_class_embeddings, axis=0)       # (num_classes, D)
        print(f"[text] Text centroids shape: {result.shape}")
        return result

    def get_logit_scale(self) -> float:
        """Return the learned temperature scaling factor."""
        return self.model.logit_scale.exp().item()


class DINOv2Extractor(BaseExtractor):
    """
    Extractor for DINOv2 models via the timm library.

    Install: pip install timm
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError:
            raise ImportError("DINOv2 extractor requires: pip install timm")

        # Default to the DINOv2 large model; DINOv3 passes its own timm name.
        model_name = self.config["model"].get("pretrained", "vit_large_patch14_dinov2.lvd142m")
        print(f"[model] Loading timm model: {model_name}")

        # Optional timm overrides. Both default to the checkpoint's own setting,
        # so omitting them reproduces the previous behaviour exactly.
        #
        #   global_pool -- how the per-patch tokens collapse into one vector.
        #       'token' = the CLS token, a learned summary that can attend
        #                 selectively and ignore background.
        #       'avg'   = the mean of the patch tokens, where every patch votes
        #                 equally, background included.
        #       timm defaults DINOv2 to 'token' and DINOv3 to 'avg', which is the
        #       single biggest behavioural difference between the two families.
        #       Measured on 10 CUB classes (543 images, k-means on raw features):
        #           full frames  token 0.535 vs avg 0.439 ARI  -> use 'token'
        #           bbox crops   avg   0.512 vs token 0.471 ARI -> use 'avg'
        #       The two do not stack: cropping removes the background that CLS
        #       pooling exists to suppress, so pick one based on how much of the
        #       frame the subject fills.
        #
        #   img_size    -- run the ViT at a different resolution. DINOv3
        #       ViT-H+/16 sees 256px = a 16x16 grid = 256 patches, against
        #       DINOv2 ViT-L/14 at 518px = 1369 patches. More patches means more
        #       spatial detail for fine-grained subjects, at quadratic attention
        #       cost. Needs a ViT with interpolatable position embeddings.
        img_size    = self.config["model"].get("img_size", None)
        global_pool = self.config["model"].get("global_pool", None)

        create_kwargs = {}
        if img_size:
            create_kwargs["img_size"] = int(img_size)
        if global_pool:
            create_kwargs["global_pool"] = global_pool

        self.model = timm.create_model(model_name, pretrained=True,
                                       num_classes=0, **create_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Let timm derive the correct input size and normalization from the model.
        # vit_large_patch14_dinov2.lvd142m expects 518x518, not 224x224.
        # Hardcoding 224 causes an AssertionError inside patch_embed.
        #
        # An img_size override MUST be threaded through here explicitly.
        # resolve_data_config reads the checkpoint's pretrained_cfg, which
        # create_model does not rewrite -- left alone it keeps reporting the
        # native size, the transform keeps emitting native-size tensors, and the
        # override silently does nothing at all.
        args = {"input_size": (3, int(img_size), int(img_size))} if img_size else {}
        data_config     = resolve_data_config(args, model=self.model)
        self.preprocess = create_transform(**data_config)

        # Store input spatial size for the error-fallback placeholder in extract().
        # data_config["input_size"] is (C, H, W), e.g. (3, 518, 518).
        self._input_size = tuple(data_config["input_size"])

        print(f"[model] {model_name} ready on device: {self.device}  "
              f"(input_size={self._input_size}, "
              f"pool={getattr(self.model, 'global_pool', '?')})")

    def extract(self, image_paths: List[str], progress_fn=None, cancelled_fn=None) -> np.ndarray:
        all_features = []
        n_failed = 0
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"[extract] {len(image_paths)} images, "
              f"batch_size={self.batch_size}, {n_batches} batches")

        _t0 = time.time()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        for batch_num, i in enumerate(tqdm(range(0, len(image_paths), self.batch_size),
                                           desc="Extracting features")):
            if cancelled_fn and cancelled_fn():
                return None

            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    n_failed += 1
                    print(f"[warn] Could not load {path}: {e} - using zeros")
                    images.append(torch.zeros(self._input_size))

            batch_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                if self.normalize:
                    features = self._normalize(features)

            all_features.append(features.cpu().numpy())
            if progress_fn:
                progress_fn(batch_num + 1, n_batches)

        if n_failed > 0:
            print(f"[warn] {n_failed}/{len(image_paths)} images failed to load")

        result = np.vstack(all_features)
        print(f"[extract] Done: {result.shape[0]} embeddings, dim={result.shape[1]}")
        _perf_summary("extract", result.shape[0], _t0)

        # Validate: no NaN values should be present
        n_nan_rows = int(np.any(np.isnan(result), axis=1).sum())
        if n_nan_rows > 0:
            raise RuntimeError(
                f"NaN detected in {n_nan_rows} feature vectors after extraction. "
                f"Run filter_valid_images() before extraction to remove bad files."
            )

        return result


# -------------------------------------------------------
# Factory: Build the Right Extractor from Config
# -------------------------------------------------------

def build_extractor(config: dict) -> BaseExtractor:
    """
    Instantiate and return the correct extractor based on config.model.name.

    Supported values:
        bioclip2, bioclip, clip  ->  OpenCLIPExtractor
        dinov2                   ->  DINOv2Extractor
    """
    name = config["model"]["name"].lower()

    if name in ("bioclip2.5", "bioclip2", "bioclip", "clip"):
        return OpenCLIPExtractor(config)
    elif name in ("dinov2", "dinov3"):
        return DINOv2Extractor(config)
    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            "Supported: bioclip2.5, bioclip2, bioclip, clip, dinov2, dinov3"
        )
