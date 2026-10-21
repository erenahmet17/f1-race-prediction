"""FIA final start gridi için doğrulama ve model özellikleri."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import numpy as np
import pandas as pd


FINAL_DOCUMENT_TYPE = "FINAL_STARTING_GRID"
REQUIRED_GRID_COLUMNS = {"DriverId", "GridPosition", "StartsFromPitLane"}


class FinalGridNotReadyError(RuntimeError):
    """FIA final start gridi henüz doğrulanmadığında oluşur."""


class GridDataError(ValueError):
    """Final grid içeriği eksik veya tutarsızsa oluşur."""


@dataclass(frozen=True)
class GridDocumentMetadata:
    """FIA grid belgesinin doğrulanmış kimliği."""

    document_type: str
    source: str
    published_at: pd.Timestamp
    document_number: str | None = None


def _require_final_document(metadata: GridDocumentMetadata) -> None:
    if metadata.document_type.strip().upper() != FINAL_DOCUMENT_TYPE:
        raise FinalGridNotReadyError(
            "FIA Final Starting Grid henüz doğrulanmadı; tahmin oluşturulamaz."
        )
    if not metadata.source.strip():
        raise GridDataError("Final grid kaynağı boş olamaz")
    if pd.isna(pd.Timestamp(metadata.published_at)):
        raise GridDataError("Final grid yayın zamanı geçersiz")


def _as_boolean(values: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    converted = values.map(lambda value: mapping.get(value, mapping.get(str(value).lower())))
    if converted.isna().any():
        raise GridDataError("StartsFromPitLane yalnızca doğru/yanlış olabilir")
    return converted.astype(bool)


def build_final_grid_features(
    grid: pd.DataFrame,
    metadata: GridDocumentMetadata,
    *,
    expected_driver_ids: Collection[str] | None = None,
) -> pd.DataFrame:
    """Yalnızca FIA final belgesiyle doğrulanmış start sırasını üretir."""
    _require_final_document(metadata)
    missing = sorted(REQUIRED_GRID_COLUMNS.difference(grid.columns))
    if missing:
        raise GridDataError(f"Final grid eksik sütunlar içeriyor: {', '.join(missing)}")
    if grid.empty:
        raise GridDataError("Final grid boş")

    features = grid.copy().reset_index(drop=True)
    features["SourceOrder"] = np.arange(len(features))
    if features["DriverId"].isna().any() or (features["DriverId"].astype(str) == "").any():
        raise GridDataError("Final gridde DriverId boş olamaz")
    if features["DriverId"].duplicated().any():
        raise GridDataError("Final gridde aynı pilot birden fazla kez bulunuyor")

    if expected_driver_ids is not None:
        expected = set(expected_driver_ids)
        observed = set(features["DriverId"])
        if expected != observed:
            missing_drivers = sorted(expected - observed)
            extra_drivers = sorted(observed - expected)
            raise GridDataError(
                "Final grid pilot listesi qualifying ile uyuşmuyor. "
                f"Eksik: {missing_drivers}; Fazla: {extra_drivers}"
            )

    features["StartsFromPitLane"] = _as_boolean(features["StartsFromPitLane"])
    features["GridPosition"] = pd.to_numeric(features["GridPosition"], errors="coerce")
    on_grid = ~features["StartsFromPitLane"]
    if features.loc[on_grid, "GridPosition"].isna().any():
        raise GridDataError("Gridde başlayan pilotların GridPosition değeri olmalı")
    if (features.loc[on_grid, "GridPosition"] <= 0).any():
        raise GridDataError("GridPosition pozitif olmalı")
    if features.loc[on_grid, "GridPosition"].duplicated().any():
        raise GridDataError("Final gridde tekrarlanan GridPosition var")

    features.loc[features["StartsFromPitLane"], "GridPosition"] = np.nan
    features = features.sort_values(
        ["StartsFromPitLane", "GridPosition", "SourceOrder"],
        na_position="last",
    ).reset_index(drop=True)
    features["StartOrder"] = np.arange(1, len(features) + 1)
    features["StartOrderPct"] = np.where(
        len(features) > 1,
        (features["StartOrder"] - 1) / (len(features) - 1),
        np.nan,
    )
    features["GridStatus"] = "FINAL"
    features["GridSource"] = metadata.source
    features["GridPublishedAt"] = pd.Timestamp(metadata.published_at)
    features["GridDocumentNumber"] = metadata.document_number

    return features.drop(columns="SourceOrder")

