"""Every module imports cleanly: the cheapest guard against a broken rename."""
import importlib

import pytest

CORE = [
    "biocluster",
    "biocluster.data.dataset",
    "biocluster.data.plugins",
    "biocluster.reduction.reduction",
    "biocluster.clustering.methods",
    "biocluster.clustering.density.dbscan_pipeline",
    "biocluster.clustering.density.hdbscan_pipeline",
    "biocluster.evaluation.metrics",
    "biocluster.annotation.annotator",
    "biocluster.annotation.taxonomy",
    "biocluster.annotation.gbif_media",
    "biocluster.annotation.interactive_report",
    "biocluster.annotation.gallery",
    "biocluster.annotation.review",
    "biocluster.models.extractors",
    "biocluster.utils.io",
    "biocluster.visualization.plots",
    "biocluster.cli.extract",
    "biocluster.cli.cluster",
    "biocluster.cli.evaluate",
    "biocluster.cli.annotate",
]

GUI = ["biocluster.app.theme", "biocluster.app.components",
       "biocluster.app.feature_cache"]


@pytest.mark.parametrize("name", CORE)
def test_core_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", GUI)
def test_gui_imports(name):
    pytest.importorskip("PySide6", reason="GUI extra not installed")
    importlib.import_module(name)


def test_version():
    import biocluster
    assert biocluster.__version__
