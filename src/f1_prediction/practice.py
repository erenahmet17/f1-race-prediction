"""Practice seanslarından yarış temposu özellikleri üretir."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


PRACTICE_SESSIONS = {"FP1", "FP2", "FP3"}
REQUIRED_LAP_COLUMNS = {
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "IsAccurate",
    "Deleted",
    "TrackStatus",
}


class PracticeDataError(ValueError):
    """Practice verisi beklenen yapıda değilse oluşur."""


@dataclass(frozen=True)
class PracticeFeatures:
    """Pilot ve takım seviyesindeki practice özellikleri."""

    drivers: pd.DataFrame
    teams: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PracticeDataError(f"{label} eksik sütunlar içeriyor: {', '.join(missing)}")


def _lap_seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values.dtype):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


def _median_absolute_deviation(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def clean_practice_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Temel kalite ve pist durumu kurallarını geçen practice turlarını seçer."""
    _require_columns(laps, REQUIRED_LAP_COLUMNS, "Practice tur verisi")

    clean = laps.copy()
    clean["LapTime_s"] = _lap_seconds(clean["LapTime"])
    clean["LapNumber"] = pd.to_numeric(clean["LapNumber"], errors="coerce")
    clean["TyreLife"] = pd.to_numeric(clean["TyreLife"], errors="coerce")
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
    )
    return clean.loc[mask].copy()


def _prepare_sessions(session_laps: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not session_laps:
        raise PracticeDataError("En az bir practice seansı gerekli")

    frames: list[pd.DataFrame] = []
    for raw_code, laps in session_laps.items():
        code = raw_code.upper()
        if code not in PRACTICE_SESSIONS:
            raise PracticeDataError(f"Practice olmayan seans verildi: {raw_code}")
        _require_columns(laps, REQUIRED_LAP_COLUMNS, code)
        frame = laps.copy()
        frame["SessionCode"] = code
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _tyre_age_bucket(tyre_life: pd.Series) -> pd.Series:
    conditions = [
        tyre_life.between(0, 5, inclusive="both"),
        tyre_life.between(5, 10, inclusive="right"),
        tyre_life.between(10, 15, inclusive="right"),
        tyre_life > 15,
    ]
    return pd.Series(
        np.select(conditions, ["01-05", "06-10", "11-15", "16+"], default="UNKNOWN"),
        index=tyre_life.index,
        dtype="string",
    )


def _eligible_long_run_laps(
    clean: pd.DataFrame, minimum_stint_laps: int
) -> pd.DataFrame:
    work = clean.copy()
    compounds = work["Compound"].fillna("UNKNOWN").astype(str).str.upper()
    work["SurfaceCondition"] = np.where(
        compounds.isin({"INTERMEDIATE", "WET"}), "WET", "DRY"
    )
    # Practice stintleri hızlı tur ve soğutma turlarını dönüşümlü içerebilir.
    # Seansın en hızlı turundan %10'dan daha yavaş turlar yarış temposunu temsil
    # etmeyen trafik/soğutma turları olarak ilk aşamada çıkarılır.
    work["SessionFastest_s"] = work.groupby(
        ["SessionCode", "SurfaceCondition"]
    )["LapTime_s"].transform("min")
    work = work[
        work["LapTime_s"] <= 1.10 * work["SessionFastest_s"]
    ].copy()
    work["StintKey"] = (
        work["SessionCode"].astype(str)
        + "|"
        + work["Driver"].astype(str)
        + "|"
        + work["Stint"].astype(str)
    )
    work["StintCleanLapCount"] = work.groupby("StintKey")["LapNumber"].transform(
        "count"
    )
    work = work[work["StintCleanLapCount"] >= minimum_stint_laps].copy()
    if work.empty:
        return work

    work["StintMedian_s"] = work.groupby("StintKey")["LapTime_s"].transform(
        "median"
    )
    work["StintMad_s"] = work.groupby("StintKey")["LapTime_s"].transform(
        _median_absolute_deviation
    )
    slow_limit = work["StintMedian_s"] + np.maximum(
        3 * work["StintMad_s"],
        0.03 * work["StintMedian_s"],
    )
    work = work[work["LapTime_s"] <= slow_limit].copy()

    remaining = work.groupby("StintKey")["LapNumber"].transform("count")
    return work[remaining >= minimum_stint_laps].copy()


def _add_relative_pace(laps: pd.DataFrame) -> pd.DataFrame:
    work = laps.copy()
    work["TyreAgeBucket"] = _tyre_age_bucket(work["TyreLife"])
    work["Compound"] = work["Compound"].fillna("UNKNOWN").astype(str)

    detailed_keys = [
        "SessionCode",
        "SurfaceCondition",
        "Compound",
        "TyreAgeBucket",
    ]
    compound_keys = ["SessionCode", "SurfaceCondition", "Compound"]

    work["DetailedReference_s"] = work.groupby(detailed_keys)[
        "LapTime_s"
    ].transform("median")
    work["DetailedDriverCount"] = work.groupby(detailed_keys)["Driver"].transform(
        "nunique"
    )
    work["CompoundReference_s"] = work.groupby(compound_keys)[
        "LapTime_s"
    ].transform("median")
    work["CompoundDriverCount"] = work.groupby(compound_keys)["Driver"].transform(
        "nunique"
    )
    work["SessionReference_s"] = work.groupby(
        ["SessionCode", "SurfaceCondition"]
    )["LapTime_s"].transform("median")

    detailed_ok = work["DetailedDriverCount"] >= 3
    compound_ok = work["CompoundDriverCount"] >= 3
    work["PracticeReference_s"] = np.select(
        [detailed_ok, compound_ok],
        [work["DetailedReference_s"], work["CompoundReference_s"]],
        default=work["SessionReference_s"],
    )
    work["PracticeReferenceFallbackLevel"] = np.select(
        [detailed_ok, compound_ok],
        [0, 1],
        default=2,
    )
    work["PracticeLapDeltaPct"] = (
        (work["LapTime_s"] - work["PracticeReference_s"])
        / work["PracticeReference_s"]
        * 100
    )
    return work


def build_practice_features(
    session_laps: Mapping[str, pd.DataFrame],
    *,
    expected_sessions: int | None = None,
    minimum_stint_laps: int = 5,
) -> PracticeFeatures:
    """FP1-FP3 uzun sürüşlerinden pilot ve takım yarış temposu üretir."""
    if minimum_stint_laps < 2:
        raise ValueError("minimum_stint_laps en az 2 olmalı")

    all_laps = _prepare_sessions(session_laps)
    participant_rows = all_laps[["Driver", "Team"]].drop_duplicates("Driver")
    team_rows = all_laps[["Team"]].drop_duplicates()
    if expected_sessions is None:
        expected_sessions = len(session_laps)
    if expected_sessions < 1:
        raise ValueError("expected_sessions en az 1 olmalı")

    clean = clean_practice_laps(all_laps)
    long_run = _eligible_long_run_laps(clean, minimum_stint_laps)

    if long_run.empty:
        driver_features = participant_rows.copy()
        driver_features["PracticePaceDeltaPct"] = np.nan
        driver_features["PracticeConsistencyPct"] = np.nan
        driver_features["PracticeLongRunLapCount"] = 0
        driver_features["PracticeStintCount"] = 0
        driver_features["PracticeSessionsUsed"] = 0
        driver_features["PracticeReferenceFallbackPct"] = np.nan
    else:
        relative = _add_relative_pace(long_run)
        relative["ReferenceWasFallback"] = (
            relative["PracticeReferenceFallbackLevel"] > 0
        )
        calculated = (
            relative.groupby(["Driver", "Team"], as_index=False)
            .agg(
                PracticePaceDeltaPct=("PracticeLapDeltaPct", "median"),
                PracticeConsistencyPct=(
                    "PracticeLapDeltaPct",
                    _median_absolute_deviation,
                ),
                PracticeLongRunLapCount=("LapNumber", "count"),
                PracticeStintCount=("StintKey", "nunique"),
                PracticeSessionsUsed=("SessionCode", "nunique"),
                PracticeReferenceFallbackPct=("ReferenceWasFallback", "mean"),
            )
        )
        driver_features = participant_rows.merge(
            calculated, on=["Driver", "Team"], how="left"
        )
        for column in (
            "PracticeLongRunLapCount",
            "PracticeStintCount",
            "PracticeSessionsUsed",
        ):
            driver_features[column] = driver_features[column].fillna(0).astype(int)

    target_laps = minimum_stint_laps * expected_sessions
    lap_score = (driver_features["PracticeLongRunLapCount"] / target_laps).clip(
        upper=1.0
    )
    session_score = (driver_features["PracticeSessionsUsed"] / expected_sessions).clip(
        upper=1.0
    )
    driver_features["PracticeDataReliability"] = 0.75 * lap_score + 0.25 * session_score
    driver_features["PracticeDataQuality"] = np.select(
        [
            driver_features["PracticeLongRunLapCount"] < minimum_stint_laps,
            driver_features["PracticeDataReliability"] < 0.60,
        ],
        ["MISSING", "LOW"],
        default="USABLE",
    )
    driver_features["PracticeDataMissing"] = (
        driver_features["PracticeDataQuality"] == "MISSING"
    )

    valid_drivers = driver_features[driver_features["PracticePaceDeltaPct"].notna()]
    if valid_drivers.empty:
        team_features = team_rows.copy()
        team_features["TeamPracticePaceDeltaPct"] = np.nan
        team_features["TeamPracticeDriverCount"] = 0
        team_features["TeamPracticeLongRunLapCount"] = 0
        team_features["TeamPracticeDataReliability"] = 0.0
    else:
        calculated_teams = (
            valid_drivers.groupby("Team", as_index=False)
            .agg(
                TeamPracticePaceDeltaPct=("PracticePaceDeltaPct", "median"),
                TeamPracticeDriverCount=("Driver", "nunique"),
                TeamPracticeLongRunLapCount=("PracticeLongRunLapCount", "sum"),
                TeamPracticeDataReliability=("PracticeDataReliability", "mean"),
            )
        )
        team_features = team_rows.merge(calculated_teams, on="Team", how="left")
        for column in ("TeamPracticeDriverCount", "TeamPracticeLongRunLapCount"):
            team_features[column] = team_features[column].fillna(0).astype(int)
        team_features["TeamPracticeDataReliability"] = team_features[
            "TeamPracticeDataReliability"
        ].fillna(0.0)

    return PracticeFeatures(
        drivers=driver_features.sort_values("Driver").reset_index(drop=True),
        teams=team_features.sort_values("Team").reset_index(drop=True),
    )
