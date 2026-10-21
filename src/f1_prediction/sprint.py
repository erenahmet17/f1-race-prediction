"""Sprint seansından ana yarış öncesi özellikler üretir."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


REQUIRED_LAP_COLUMNS = {
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "PitOutTime",
    "PitInTime",
    "IsAccurate",
    "Deleted",
    "TrackStatus",
}


class SprintDataError(ValueError):
    """Sprint verisi beklenen yapıda değilse oluşur."""


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SprintDataError(f"{label} eksik sütunlar içeriyor: {', '.join(missing)}")


def _lap_seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values.dtype):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


def clean_sprint_laps(laps: pd.DataFrame, opening_laps: int = 2) -> pd.DataFrame:
    """Karşılaştırılabilir yeşil bayrak sprint turlarını seçer."""
    _require_columns(laps, REQUIRED_LAP_COLUMNS, "Sprint tur verisi")

    clean = laps.copy()
    clean["LapTime_s"] = _lap_seconds(clean["LapTime"])
    clean["LapNumber"] = pd.to_numeric(clean["LapNumber"], errors="coerce")
    accurate = clean["IsAccurate"].astype("boolean").fillna(False)
    # FastF1, tur silinmemişse Deleted alanını çoğu zaman None bırakır.
    not_deleted = ~clean["Deleted"].astype("boolean").fillna(False)

    mask = (
        clean["LapTime_s"].notna()
        & clean["PitOutTime"].isna()
        & clean["PitInTime"].isna()
        & accurate
        & not_deleted
        & (clean["TrackStatus"].astype(str) == "1")
        & (clean["LapNumber"] > opening_laps)
    )
    return clean.loc[mask].copy()


def _median_absolute_deviation(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def _completed_status(status: object) -> bool:
    value = str(status).strip().lower()
    return value == "finished" or bool(re.search(r"(?:lapped|\+\s*\d+\s*lap)", value))


def _result_features(results: pd.DataFrame) -> pd.DataFrame:
    driver_column = "Abbreviation" if "Abbreviation" in results.columns else "Driver"
    required = {driver_column, "GridPosition", "Position", "Status", "Laps"}
    _require_columns(results, required, "Sprint sonuç verisi")

    columns = [driver_column, "GridPosition", "Position", "Status", "Laps"]
    if "TeamName" in results.columns:
        columns.append("TeamName")
    selected = results[columns].copy()
    selected = selected.rename(
        columns={
            driver_column: "Driver",
            "GridPosition": "SprintStartPosition",
            "Position": "SprintFinishPosition",
            "Status": "SprintStatus",
            "Laps": "SprintLapsCompleted",
            "TeamName": "ResultTeam",
        }
    )
    for column in (
        "SprintStartPosition",
        "SprintFinishPosition",
        "SprintLapsCompleted",
    ):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    max_laps = selected["SprintLapsCompleted"].max()
    selected["SprintPositionsGained"] = (
        selected["SprintStartPosition"] - selected["SprintFinishPosition"]
    )
    selected["SprintLapsCompletedPct"] = np.where(
        max_laps > 0,
        selected["SprintLapsCompleted"] / max_laps,
        np.nan,
    )
    selected["SprintDNF"] = ~selected["SprintStatus"].map(_completed_status)
    return selected


def build_sprint_features(
    laps: pd.DataFrame,
    results: pd.DataFrame | None = None,
    *,
    minimum_laps: int = 5,
    usable_laps: int = 10,
    opening_laps: int = 2,
) -> pd.DataFrame:
    """Pilot ve takım bazında ana yarış öncesi sprint özellikleri üretir.

    Her temiz tur, aynı sprint turundaki saha medyanıyla yüzde olarak
    karşılaştırılır. Beş turun altındaki tempo örnekleri modele verilmez.
    """
    if minimum_laps < 1:
        raise ValueError("minimum_laps en az 1 olmalı")
    if usable_laps < minimum_laps:
        raise ValueError("usable_laps minimum_laps değerinden küçük olamaz")

    clean = clean_sprint_laps(laps, opening_laps=opening_laps)
    if clean.empty:
        features = pd.DataFrame(
            columns=[
                "Driver",
                "Team",
                "SprintPaceDeltaPct",
                "SprintConsistencyPct",
                "SprintCleanLapCount",
                "SprintDataReliability",
                "SprintDataQuality",
            ]
        )
    else:
        clean["FieldMedian_s"] = clean.groupby("LapNumber")[
            "LapTime_s"
        ].transform("median")
        clean["SprintLapDeltaPct"] = (
            (clean["LapTime_s"] - clean["FieldMedian_s"])
            / clean["FieldMedian_s"]
            * 100
        )

        features = (
            clean.groupby(["Driver", "Team"], as_index=False)
            .agg(
                SprintPaceDeltaPct=("SprintLapDeltaPct", "median"),
                SprintConsistencyPct=(
                    "SprintLapDeltaPct",
                    _median_absolute_deviation,
                ),
                SprintCleanLapCount=("LapNumber", "count"),
            )
            .sort_values("Driver")
            .reset_index(drop=True)
        )

        reference_laps = int(features["SprintCleanLapCount"].max())
        features["SprintDataReliability"] = (
            features["SprintCleanLapCount"] / reference_laps
        ).clip(upper=1.0)
        features["SprintDataQuality"] = np.select(
            [
                features["SprintCleanLapCount"] < minimum_laps,
                features["SprintCleanLapCount"] < usable_laps,
            ],
            ["MISSING", "LOW"],
            default="USABLE",
        )

        too_short = features["SprintCleanLapCount"] < minimum_laps
        features.loc[
            too_short, ["SprintPaceDeltaPct", "SprintConsistencyPct"]
        ] = np.nan

    if results is not None:
        features = features.merge(_result_features(results), on="Driver", how="outer")
        if "ResultTeam" in features.columns:
            features["Team"] = features["Team"].fillna(features["ResultTeam"])
            features = features.drop(columns="ResultTeam")

    if "SprintCleanLapCount" in features.columns:
        features["SprintCleanLapCount"] = features["SprintCleanLapCount"].fillna(0).astype(int)
        features["SprintDataReliability"] = features[
            "SprintDataReliability"
        ].fillna(0.0)
        features["SprintDataQuality"] = features["SprintDataQuality"].fillna(
            "MISSING"
        )

    team_pace = (
        features.groupby("Team", as_index=False)["SprintPaceDeltaPct"]
        .median()
        .rename(columns={"SprintPaceDeltaPct": "TeamSprintPaceDeltaPct"})
    )
    features = features.merge(team_pace, on="Team", how="left")

    return features.sort_values("Driver").reset_index(drop=True)
