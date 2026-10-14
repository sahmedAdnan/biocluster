"""
app/credits.py
--------------
Who made Bio Cluster and what it is built on, for the desktop app's
"About & credits" window. Plain data, kept apart from the Qt code so names,
affiliations and links live in one place.

Keep in step with CITATION.cff, the source of truth for citations.
"""

TAGLINE = ("A framework for clustering and zero-shot labeling of ecological "
           "images using foundation models")

DEVELOPERS = ["Shakil Ahmed", "Kim Bjerge"]

AUTHORS = [
    {"name": "Shakil Ahmed",
     "affiliation": "Department of Computer Science, Aarhus University",
     "orcid": "0000-0001-6366-4881"},
    {"name": "Kim Bjerge",
     "affiliation": "Department of Electrical and Computer Engineering, Aarhus University",
     "orcid": "0000-0001-6742-9504"},
    {"name": "Toke T. Høye",
     "affiliation": "Department of Ecoscience, Aarhus University",
     "orcid": "0000-0001-5387-3284"},
    {"name": "Christian Nørgaard Storm Pedersen",
     "affiliation": "Department of Molecular Biology and Genetics, Aarhus University",
     "orcid": None},
]

# The foundation models the Model dropdown offers: (name, maker, link to the
# checkpoint the app downloads).
BUILT_ON = [
    ("BioCLIP", "Imageomics Institute", "https://huggingface.co/imageomics/bioclip"),
    ("BioCLIP 2", "Imageomics Institute", "https://huggingface.co/imageomics/bioclip-2"),
    ("BioCLIP 2.5", "Imageomics Institute", "https://huggingface.co/imageomics/bioclip-2.5-vith14"),
    ("CLIP", "OpenAI", "https://github.com/openai/CLIP"),
    ("DINOv2", "Meta AI", "https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m"),
    ("DINOv3", "Meta AI", "https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m"),
]

REPOSITORY = "https://github.com/sahmedAdnan/biocluster"
LICENSE = "MIT License · © 2026 Shakil Ahmed"
