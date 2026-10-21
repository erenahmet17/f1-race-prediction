"""Yarış öncesi oturum özelliklerini tek pilot tablosunda birleştirir."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd


class WeekendFeatureError(ValueError):
    """Bir yarış haftasının özellikleri güvenle birleştirilemiyorsa oluşur."""


QUALIFYING_COLUMNS = {
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
}
GRID_REQUIRED_COLUMNS = {
    "DriverId",
    "GridPosition",
    "StartsFromPitLane",
    "StartOrder",
    "StartOrderPct",
    "GridStatus",
    "GridSource",
    "GridPublishedAt",
    "GridDocumentNumber",
}
PRACTICE_DRIVER_COLUMNS = {
    "Driver",
    "Team",
    "PracticePaceDeltaPct",
    "PracticeConsistencyPct",
    "PracticeLongRunLapCount",
    "PracticeStintCount",
    "PracticeSessionsUsed",
    "PracticeReferenceFallbackPct",
    "PracticeDataReliability",
    "PracticeDataQuality",
    "PracticeDataMissing",
}
PRACTICE_TEAM_COLUMNS = {
    "Team",
    "TeamPracticePaceDeltaPct",
    "TeamPracticeDriverCount",
    "TeamPracticeLongRunLapCount",
    "TeamPracticeDataReliability",
}
SPRINT_COLUMNS = {
    "Driver",
    "Team",
    "SprintPaceDeltaPct",
    "SprintConsistencyPct",
    "SprintCleanLapCount",
    "SprintDataReliability",
    "SprintDataQuality",
    "SprintStartPosition",
    "SprintFinishPosition",
    "SprintStatus",
    "SprintLapsCompleted",
    "SprintPositionsGained",
    "SprintLapsCompletedPct",
    "SprintDNF",
    "TeamSprintPaceDeltaPct",
}
GRID_PENALTY_BOOLEAN_COLUMNS = {
    "GridPenaltyNotePresent",
    "GridPenaltyApplied",
    "GridPenaltyPowerUnit",
    "GridPenaltySporting",
    "GridPenaltyParcFerme",
    "GridPenaltyTechnical",
    "GridPenaltyPitLane",
    "GridPermissionToStart",
    "GridPenaltyOther",
}


def _require_columns(
    frame: pd.DataFrame, required: Collection[str], label: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise WeekendFeatureError(
            f"{label} eksik sütunlar içeriyor: {', '.join(missing)}"
        )


def _require_unique(frame: pd.DataFrame, key: str, label: str) -> None:
    if frame[key].isna().any() or (frame[key].astype(str).str.strip() == "").any():
        raise WeekendFeatureError(f"{label} içinde {key} boş olamaz")
    duplicates = sorted(frame.loc[frame[key].duplicated(), key].astype(str).unique())
    if duplicates:
        raise WeekendFeatureError(
            f"{label} içinde tekrarlanan {key} var: {', '.join(duplicates)}"
        )


def _require_no_feature_collisions(
    base: pd.DataFrame,
    incoming: pd.DataFrame,
    join_columns: Collection[str],
    label: str,
) -> None:
    collisions = sorted(
        set(base.columns).intersection(incoming.columns).difference(join_columns)
    )
    if collisions:
        raise WeekendFeatureError(
            f"{label} sütunları mevcut tabloyla çakışıyor: {', '.join(collisions)}"
        )


def _merge_driver_features(
    base: pd.DataFrame,
    features: pd.DataFrame | None,
    *,
    required: Collection[str],
    label: str,
    available_column: str,
) -> pd.DataFrame:
    if features is None or features.empty:
        result = base.copy()
        result[available_column] = False
        return result

    _require_columns(features, required, label)
    _require_unique(features, "Driver", label)
    _require_no_feature_collisions(base, features, {"Driver", "Team"}, label)

    prepared = features.copy().rename(columns={"Team": f"{label}Team"})
    prepared[available_column] = True
    result = base.merge(prepared, on="Driver", how="left", validate="one_to_one")

    source_team = f"{label}Team"
    team_mismatch = result[source_team].notna() & (
        result[source_team].astype(str) != result["Team"].astype(str)
    )
    if team_mismatch.any():
        drivers = sorted(result.loc[team_mismatch, "Driver"].astype(str))
        raise WeekendFeatureError(
            f"{label} takım bilgisi qualifying ile uyuşmuyor: {', '.join(drivers)}"
        )
    result[available_column] = result[available_column].eq(True)
    return result.drop(columns=source_team)


def _merge_team_features(
    base: pd.DataFrame, features: pd.DataFrame | None
) -> pd.DataFrame:
    available_column = "PracticeTeamDataAvailable"
    if features is None or features.empty:
        result = base.copy()
        result[available_column] = False
        return result

    _require_columns(features, PRACTICE_TEAM_COLUMNS, "Practice takım özellikleri")
    _require_unique(features, "Team", "Practice takım özellikleri")
    _require_no_feature_collisions(
        base, features, {"Team"}, "Practice takım özellikleri"
    )
    prepared = features.copy()
    prepared[available_column] = True
    result = base.merge(prepared, on="Team", how="left", validate="many_to_one")
    result[available_column] = result[available_column].eq(True)
    return result


def _validate_grid(qualifying: pd.DataFrame, grid: pd.DataFrame) -> None:
    _require_columns(grid, GRID_REQUIRED_COLUMNS, "FIA final grid özellikleri")
    _require_unique(grid, "DriverId", "FIA final grid özellikleri")
    if not grid["GridStatus"].astype(str).str.upper().eq("FINAL").all():
        raise WeekendFeatureError("Yalnızca FIA FINAL grid özellikleri kullanılabilir")

    expected = set(qualifying["DriverId"])
    observed = set(grid["DriverId"])
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise WeekendFeatureError(
            "Final grid pilot listesi qualifying ile uyuşmuyor. "
            f"Eksik: {missing}; Fazla: {extra}"
        )


def _fill_practice_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    for column in (PRACTICE_DRIVER_COLUMNS | PRACTICE_TEAM_COLUMNS) - {
        "Driver",
        "Team",
    }:
        if column not in result:
            result[column] = pd.NA

    integer_columns = (
        "PracticeLongRunLapCount",
        "PracticeStintCount",
        "PracticeSessionsUsed",
        "TeamPracticeDriverCount",
        "TeamPracticeLongRunLapCount",
    )
    for column in integer_columns:
        if column not in result:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)

    for column in ("PracticeDataReliability", "TeamPracticeDataReliability"):
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    if "PracticeDataQuality" not in result:
        result["PracticeDataQuality"] = "MISSING"
    result["PracticeDataQuality"] = result["PracticeDataQuality"].fillna("MISSING")
    if "PracticeDataMissing" not in result:
        result["PracticeDataMissing"] = True
    result["PracticeDataMissing"] = (
        result["PracticeDataMissing"].astype("boolean").fillna(True).astype(bool)
    )
    result["TeamPracticeDataMissing"] = (
        ~result["PracticeTeamDataAvailable"]
        | result["TeamPracticePaceDeltaPct"].isna()
    )
    return result


def _fill_sprint_missing_values(
    features: pd.DataFrame, *, is_sprint_weekend: bool
) -> pd.DataFrame:
    result = features.copy()
    for column in SPRINT_COLUMNS - {"Driver", "Team"}:
        if column not in result:
            result[column] = pd.NA

    if "SprintCleanLapCount" not in result:
        result["SprintCleanLapCount"] = 0
    result["SprintCleanLapCount"] = (
        pd.to_numeric(result["SprintCleanLapCount"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    if "SprintDataReliability" not in result:
        result["SprintDataReliability"] = 0.0
    result["SprintDataReliability"] = pd.to_numeric(
        result["SprintDataReliability"], errors="coerce"
    ).fillna(0.0)
    if "SprintDataQuality" not in result:
        result["SprintDataQuality"] = pd.NA
    missing_label = "MISSING" if is_sprint_weekend else "NOT_APPLICABLE"
    result["SprintDataQuality"] = result["SprintDataQuality"].fillna(missing_label)
    result["SprintWeekend"] = bool(is_sprint_weekend)
    result["SprintDataMissing"] = (
        result["SprintDataQuality"].eq("MISSING") if is_sprint_weekend else False
    )
    return result


def build_weekend_feature_table(
    qualifying: pd.DataFrame,
    final_grid: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    event_name: str,
    practice_drivers: pd.DataFrame | None = None,
    practice_teams: pd.DataFrame | None = None,
    sprint: pd.DataFrame | None = None,
    is_sprint_weekend: bool = False,
) -> pd.DataFrame:
    """Yarış başlamadan bilinen özellikleri pilot başına tek satırda birleştirir."""
    if season < 1950:
        raise ValueError("season geçerli bir F1 sezonu olmalı")
    if round_number < 1:
        raise ValueError("round_number en az 1 olmalı")
    if not event_name.strip():
        raise ValueError("event_name boş olamaz")

    _require_columns(qualifying, QUALIFYING_COLUMNS, "Qualifying özellikleri")
    if qualifying.empty:
        raise WeekendFeatureError("Qualifying özellikleri boş")
    _require_unique(qualifying, "DriverId", "Qualifying özellikleri")
    _require_unique(qualifying, "Driver", "Qualifying özellikleri")
    _validate_grid(qualifying, final_grid)

    if is_sprint_weekend and (sprint is None or sprint.empty):
        raise WeekendFeatureError("Sprint hafta sonunda sprint özellikleri gerekli")
    if not is_sprint_weekend and sprint is not None and not sprint.empty:
        raise WeekendFeatureError("Normal yarış haftasına sprint özellikleri verilemez")

    _require_no_feature_collisions(
        qualifying, final_grid, {"DriverId"}, "FIA final grid özellikleri"
    )
    combined = qualifying.merge(
        final_grid, on="DriverId", how="inner", validate="one_to_one"
    )
    for column in GRID_PENALTY_BOOLEAN_COLUMNS:
        if column not in combined:
            combined[column] = False
        combined[column] = combined[column].astype("boolean").fillna(False).astype(bool)
    if "GridPenaltyPositions" not in combined:
        combined["GridPenaltyPositions"] = pd.NA
    if "GridPenaltyReason" not in combined:
        combined["GridPenaltyReason"] = "NONE"
    if "GridPenaltyText" not in combined:
        combined["GridPenaltyText"] = ""
    combined["GridPenaltyReason"] = combined["GridPenaltyReason"].fillna("NONE")
    combined["GridPenaltyText"] = combined["GridPenaltyText"].fillna("")
    combined["GridVsQualifyingDelta"] = (
        pd.to_numeric(combined["StartOrder"], errors="coerce")
        - pd.to_numeric(combined["QualifyingPosition"], errors="coerce")
    )
    combined["GridVsQualifyingDeltaPct"] = (
        pd.to_numeric(combined["StartOrderPct"], errors="coerce")
        - pd.to_numeric(combined["QualifyingPositionPct"], errors="coerce")
    )
    combined = _merge_driver_features(
        combined,
        practice_drivers,
        required=PRACTICE_DRIVER_COLUMNS,
        label="Practice",
        available_column="PracticeDriverDataAvailable",
    )
    combined = _merge_team_features(combined, practice_teams)
    combined = _merge_driver_features(
        combined,
        sprint,
        required=SPRINT_COLUMNS,
        label="Sprint",
        available_column="SprintDataAvailable",
    )

    combined = _fill_practice_missing_values(combined)
    combined = _fill_sprint_missing_values(
        combined, is_sprint_weekend=is_sprint_weekend
    )
    combined["PreRaceDataMissing"] = (
        combined["PracticeDataMissing"]
        | combined["TeamPracticeDataMissing"]
        | combined["QualiTimeMissing"]
        | (combined["SprintDataMissing"] if is_sprint_weekend else False)
    )

    combined.insert(0, "EventName", event_name.strip())
    combined.insert(0, "RoundNumber", int(round_number))
    combined.insert(0, "Season", int(season))
    return combined.sort_values("StartOrder").reset_index(drop=True)
