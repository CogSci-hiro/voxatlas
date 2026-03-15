"""Sphinx configuration for VoxAtlas documentation."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))


class _TypeExpr:
    """Stand-in object for mocked runtime types used in annotations."""

    def __init__(self, name: str) -> None:
        self.__name__ = name

    def __or__(self, other: object) -> "_TypeExpr":
        other_name = getattr(other, "__name__", repr(other))
        return _TypeExpr(f"{self.__name__} | {other_name}")

    def __ror__(self, other: object) -> "_TypeExpr":
        other_name = getattr(other, "__name__", repr(other))
        return _TypeExpr(f"{other_name} | {self.__name__}")

    def __getitem__(self, item: object) -> "_TypeExpr":
        return self

    def __repr__(self) -> str:
        return self.__name__


def _ensure_mock_module(name: str, attrs: dict[str, object] | None = None) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module

    for attr_name, value in (attrs or {}).items():
        setattr(module, attr_name, value)


def _install_doc_mocks() -> None:
    dataframe = _TypeExpr("DataFrame")
    series = _TypeExpr("Series")
    ndarray = _TypeExpr("ndarray")

    _ensure_mock_module("numpy", {"ndarray": ndarray, "float64": _TypeExpr("float64")})
    _ensure_mock_module("pandas", {"DataFrame": dataframe, "Series": series})
    _ensure_mock_module("yaml")
    _ensure_mock_module("soundfile")
    _ensure_mock_module("parselmouth")
    _ensure_mock_module("spacy")
    _ensure_mock_module("stanza")
    _ensure_mock_module("moviepy")
    _ensure_mock_module("moviepy.editor", {"VideoFileClip": _TypeExpr("VideoFileClip")})
    _ensure_mock_module("scipy")
    _ensure_mock_module(
        "scipy.signal",
        {
            "correlate": lambda *args, **kwargs: None,
            "find_peaks": lambda *args, **kwargs: (None, {}),
            "hilbert": lambda *args, **kwargs: None,
            "stft": lambda *args, **kwargs: (None, None, None),
        },
    )


_install_doc_mocks()

project = "VoxAtlas"
author = "VoxAtlas contributors"
copyright = "2026, VoxAtlas contributors"
release = os.environ.get("VOXATLAS_DOCS_VERSION", "0.1.0")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_generate_overwrite = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"

autodoc_mock_imports = [
    "moviepy",
    "numpy",
    "pandas",
    "parselmouth",
    "scipy",
    "soundfile",
    "spacy",
    "stanza",
    "yaml",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/your-org/voxatlas",
}
