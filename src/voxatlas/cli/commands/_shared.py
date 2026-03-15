from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.core.discovery import discover_features


def ensure_feature_discovery() -> None:
    discover_features()


def extractor_description(extractor_cls: type) -> str:
    return getattr(extractor_cls, "description", None) or inspect.getdoc(extractor_cls) or ""


def format_dependencies(dependencies: tuple[str, ...] | list[str]) -> str:
    return ", ".join(dependencies) if dependencies else "-"


def sanitize_feature_name(feature_name: str) -> str:
    return feature_name.replace(".", "_")


def write_feature_output(output_dir: Path, feature_name: str, output) -> None:
    feature_dir = output_dir / sanitize_feature_name(feature_name)
    feature_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature": getattr(output, "feature", feature_name),
        "unit": getattr(output, "unit", None),
        "output_type": type(output).__name__,
    }
    (feature_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    values = getattr(output, "values", None)

    if isinstance(values, pd.Series):
        values.to_csv(feature_dir / "values.csv", index=True)
        return

    if isinstance(values, pd.DataFrame):
        values.to_csv(feature_dir / "values.csv", index=False)
        return

    if isinstance(values, np.ndarray):
        np.save(feature_dir / "values.npy", values)
        return

    if values is None:
        return

    (feature_dir / "values.txt").write_text(str(values), encoding="utf-8")
