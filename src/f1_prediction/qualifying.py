"""Ana qualifying sonuçlarını yarış öncesi model özelliklerine dönüştürür."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "DriverId",
    "Abbreviation",
    "TeamName",
    "Position",
    "Q1",
    "Q2",
    "Q3",
}


class QualifyingDataError(ValueError):
    """Qualifying verisi eksik veya tutarsızsa oluşur."""


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise QualifyingDataError(
            f"Qualifying sonucu eksik sütunlar içeriyor: {', '.join(missing)}"
        )


def _to_seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values.dtype):
        return values.dt.total_seconds()
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        if values.notna().any():
            raise QualifyingDataError("Qualifying süresi beklenmeyen tarih türünde")
        return pd.Series(np.nan, index=values.index, dtype=float)
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def _stage_reached(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["Q3_s"].notna(),
                frame["Q2_s"].notna(),
                frame["Q1_s"].notna(),
            ],
            ["Q3", "Q2", "Q1"],
            default="NO_TIME",
        ),
        index=frame.index,
        dtype="string",
    )


def build_qualifying_features(results: pd.DataFrame) -> pd.DataFrame:
    """Pilotların ana qualifying performansını pole turuna göre yüzdeleştirir."""
    _require_columns(results)
    if results.empty:
        raise QualifyingDataError("Qualifying sonucu boş")
    if results["DriverId"].isna().any():
        raise QualifyingDataError("DriverId boş olamaz")
    if results["DriverId"].duplicated().any():
        duplicates = sorted(results.loc[results["DriverId"].duplicated(), "DriverId"])
        raise QualifyingDataError(
            f"Qualifying sonucunda tekrarlanan pilot var: {', '.join(duplicates)}"
        )

    features = results[
        ["DriverId", "Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3"]
    ].copy()
    features = features.rename(
        columns={
            "Abbreviation": "Driver",
            "TeamName": "Team",
            "Position": "QualifyingPosition",
        }
    )
    features["QualifyingPosition"] = pd.to_numeric(
        features["QualifyingPosition"], errors="coerce"
    )
    for stage in ("Q1", "Q2", "Q3"):
        features[f"{stage}_s"] = _to_seconds(features[stage])

    features["QualiStageReached"] = _stage_reached(features)
    features["QualiTime_s"] = np.select(
        [
            features["Q3_s"].notna(),
            features["Q2_s"].notna(),
            features["Q1_s"].notna(),
        ],
        [features["Q3_s"], features["Q2_s"], features["Q1_s"]],
        default=np.nan,
    )
    if features["QualiTime_s"].notna().sum() == 0:
        raise QualifyingDataError("Hiçbir pilot için geçerli qualifying süresi yok")

    for stage in ("Q1", "Q2", "Q3"):
        reference = features[f"{stage}_s"].min(skipna=True)
        features[f"{stage}GapPct"] = np.where(
            pd.notna(reference) & (reference > 0),
            (features[f"{stage}_s"] - reference) / reference * 100,
            np.nan,
        )

    stage_reference = features.groupby("QualiStageReached")["QualiTime_s"].transform(
        "min"
    )
    features["QualiGapPct"] = (
        (features["QualiTime_s"] - stage_reference) / stage_reference * 100
    )
    driver_count = len(features)
    features["QualifyingPositionPct"] = np.where(
        driver_count > 1,
        (features["QualifyingPosition"] - 1) / (driver_count - 1),
        np.nan,
    )
    features["QualiTimeMissing"] = features["QualiTime_s"].isna()

    same_stage_team_count = features.groupby(["Team", "QualiStageReached"])[
        "DriverId"
    ].transform("count")
    same_stage_team_best = features.groupby(["Team", "QualiStageReached"])[
        "QualiGapPct"
    ].transform("min")
    features["QualiGapToTeammatePct"] = np.where(
        same_stage_team_count >= 2,
        features["QualiGapPct"] - same_stage_team_best,
        np.nan,
    )
    team_best_position = features.groupby("Team")["QualifyingPosition"].transform(
        "min"
    )
    features["QualifyingPositionToTeammate"] = (
        features["QualifyingPosition"] - team_best_position
    )
    features["TeamQualifyingPositionMean"] = features.groupby("Team")[
        "QualifyingPosition"
    ].transform("mean")
    features["QualifyingDataQuality"] = np.select(
        [
            features["QualiTimeMissing"],
            features["QualifyingPosition"].isna(),
        ],
        ["MISSING", "LOW"],
        default="USABLE",
    )

    return features[
        [
            "DriverId",
            "Driver",
            "Team",
            "QualifyingPosition",
            "QualifyingPositionPct",
            "QualiTime_s",
            "QualiGapPct",
            "Q1GapPct",
            "Q2GapPct",
            "Q3GapPct",
            "QualiGapToTeammatePct",
            "QualifyingPositionToTeammate",
            "TeamQualifyingPositionMean",
            "QualiStageReached",
            "QualiTimeMissing",
            "QualifyingDataQuality",
        ]
    ].sort_values("QualifyingPosition", na_position="last").reset_index(drop=True)
