"""Pist geometrisi ve geçmiş pozisyon değişimi özellikleri."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


class TrackDataError(ValueError):
    """Pist özellikleri güvenle üretilemediğinde oluşur."""


TRACK_GEOMETRY_COLUMNS = (
    "TrackId",
    "TrackName",
    "TrackLapDistanceKm",
    "TrackLapDistanceApproximate",
    "TrackCornerCount",
    "TrackCornersPerKm",
    "TrackGeometryDataQuality",
    "TrackGeometryDataMissing",
    "TrackGeometrySource",
)
TRACK_HISTORY_COLUMNS = (
    "TrackHistoricalPositionChangePct",
    "TrackHistoricalPositionChangeRelative",
    "TrackHistoricalGridRetention",
    "TrackHistoryRaceCount",
    "TrackHistoryDriverSampleCount",
    "TrackHistoryReliability",
    "TrackHistoryDataMissing",
)
HISTORY_REQUIRED_COLUMNS = {
    "Season",
    "RoundNumber",
    "TrackId",
    "StartOrder",
    "ActualPosition",
    "CleanMAEEligible",
}


def _session_circuit_identity(session: object) -> tuple[str, str]:
    try:
        circuit = session.session_info["Meeting"]["Circuit"]
        key = str(circuit["Key"]).strip()
        name = str(circuit["ShortName"]).strip()
    except (KeyError, TypeError, AttributeError) as exc:
        raise TrackDataError("FastF1 session_info içinde pist kimliği bulunamadı") from exc
    if not key or not name:
        raise TrackDataError("FastF1 pist kimliği boş olamaz")
    return key, name


def _corner_count(corners: pd.DataFrame) -> int | None:
    if corners is None or corners.empty or "Number" not in corners.columns:
        return None
    numbers = pd.to_numeric(corners["Number"], errors="coerce")
    letters = (
        corners["Letter"].fillna("").astype(str)
        if "Letter" in corners.columns
        else pd.Series("", index=corners.index)
    )
    labels = numbers.astype("Int64").astype("string") + letters.str.strip()
    labels = labels[numbers.notna()]
    return int(labels.nunique()) if not labels.empty else None


def _telemetry_lap_distance_km(session: object) -> float | None:
    try:
        fastest_lap = session.laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry(frequency="original")
    except Exception:
        return None
    if telemetry is None or telemetry.empty or "Distance" not in telemetry.columns:
        return None
    distance = pd.to_numeric(telemetry["Distance"], errors="coerce")
    if distance.notna().sum() < 2:
        return None
    length_km = float(distance.max() - distance.min()) / 1000
    if not 1.0 <= length_km <= 10.0:
        return None
    return length_km


def build_track_geometry_features(session: object) -> dict[str, object]:
    """Qualifying seansından pist kimliği, yaklaşık mesafe ve viraj sayısı üretir."""
    track_id, track_name = _session_circuit_identity(session)
    try:
        circuit_info = session.get_circuit_info()
    except Exception:
        circuit_info = None

    corner_count = (
        _corner_count(circuit_info.corners) if circuit_info is not None else None
    )
    length_km = _telemetry_lap_distance_km(session)
    if length_km is not None and corner_count is not None:
        quality = "USABLE"
    elif length_km is not None or corner_count is not None:
        quality = "PARTIAL"
    else:
        quality = "MISSING"

    return {
        "TrackId": track_id,
        "TrackName": track_name,
        "TrackLapDistanceKm": length_km,
        "TrackLapDistanceApproximate": True,
        "TrackCornerCount": corner_count,
        "TrackCornersPerKm": (
            corner_count / length_km
            if corner_count is not None and length_km is not None
            else np.nan
        ),
        "TrackGeometryDataQuality": quality,
        "TrackGeometryDataMissing": quality == "MISSING",
        "TrackGeometrySource": "FASTF1_MULTIVIEWER_AND_QUALIFYING_TELEMETRY",
    }


def _as_boolean(values: pd.Series, label: str) -> pd.Series:
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
    converted = values.map(
        lambda value: mapping.get(value, mapping.get(str(value).strip().lower()))
    )
    if converted.isna().any():
        raise TrackDataError(f"{label} doğru/yanlış olmalı")
    return converted.astype(bool)


def _eligible_history(
    history: pd.DataFrame, *, target_season: int, target_round: int
) -> pd.DataFrame:
    missing = sorted(HISTORY_REQUIRED_COLUMNS.difference(history.columns))
    if missing:
        raise TrackDataError(
            f"Pist geçmişinde eksik sütunlar var: {', '.join(missing)}"
        )
    work = history.copy()
    work["Season"] = pd.to_numeric(work["Season"], errors="coerce")
    work["RoundNumber"] = pd.to_numeric(work["RoundNumber"], errors="coerce")
    work["StartOrder"] = pd.to_numeric(work["StartOrder"], errors="coerce")
    work["ActualPosition"] = pd.to_numeric(
        work["ActualPosition"], errors="coerce"
    )
    event_keys = ["Season", "RoundNumber", "TrackId"]
    work["EventFieldSize"] = work.groupby(event_keys)["TrackId"].transform(
        "size"
    )
    work["CleanMAEEligible"] = _as_boolean(
        work["CleanMAEEligible"], "CleanMAEEligible"
    )
    before_target = (work["Season"] < target_season) | (
        (work["Season"] == target_season)
        & (work["RoundNumber"] < target_round)
    )
    return work[
        before_target
        & work["CleanMAEEligible"]
        & work["StartOrder"].notna()
        & work["ActualPosition"].notna()
    ].copy()


def _event_position_change(history: pd.DataFrame) -> pd.DataFrame:
    work = history.copy()
    event_keys = ["Season", "RoundNumber", "TrackId"]
    if "EventFieldSize" not in work.columns:
        work["EventFieldSize"] = work.groupby(event_keys)[
            ["StartOrder", "ActualPosition"]
        ].transform("max").max(axis=1)
    denominator = (work["EventFieldSize"] - 1).replace(0, np.nan)
    work["PositionChangePct"] = (
        (work["ActualPosition"] - work["StartOrder"]).abs()
        / denominator
        * 100
    )

    rows: list[dict[str, object]] = []
    for keys, event in work.groupby(event_keys, sort=False):
        if len(event) < 2:
            continue
        retention = event["StartOrder"].corr(
            event["ActualPosition"], method="spearman"
        )
        rows.append(
            {
                "Season": keys[0],
                "RoundNumber": keys[1],
                "TrackId": str(keys[2]),
                "EventPositionChangePct": event["PositionChangePct"].mean(),
                "EventGridRetention": retention,
                "EventDriverSampleCount": len(event),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["SeasonPositionChangeBaseline"] = events.groupby("Season")[
        "EventPositionChangePct"
    ].transform("median")
    events["EventPositionChangeRelative"] = np.where(
        events["SeasonPositionChangeBaseline"] > 0,
        events["EventPositionChangePct"]
        / events["SeasonPositionChangeBaseline"],
        np.nan,
    )
    return events


def empty_track_history_features() -> dict[str, object]:
    return {
        "TrackHistoricalPositionChangePct": np.nan,
        "TrackHistoricalPositionChangeRelative": np.nan,
        "TrackHistoricalGridRetention": np.nan,
        "TrackHistoryRaceCount": 0,
        "TrackHistoryDriverSampleCount": 0,
        "TrackHistoryReliability": 0.0,
        "TrackHistoryDataMissing": True,
    }


def build_track_history_features(
    history: pd.DataFrame,
    *,
    track_id: str,
    target_season: int,
    target_round: int,
) -> dict[str, object]:
    """Hedef yarıştan önceki pist hareketliliğini sezon geneline oranlar."""
    if history.empty:
        return empty_track_history_features()
    eligible = _eligible_history(
        history,
        target_season=target_season,
        target_round=target_round,
    )
    events = _event_position_change(eligible)
    if events.empty:
        return empty_track_history_features()
    track_events = events[events["TrackId"].astype(str) == str(track_id)]
    if track_events.empty:
        return empty_track_history_features()

    race_count = len(track_events)
    return {
        "TrackHistoricalPositionChangePct": float(
            track_events["EventPositionChangePct"].median()
        ),
        "TrackHistoricalPositionChangeRelative": float(
            track_events["EventPositionChangeRelative"].median()
        ),
        "TrackHistoricalGridRetention": float(
            track_events["EventGridRetention"].median()
        ),
        "TrackHistoryRaceCount": race_count,
        "TrackHistoryDriverSampleCount": int(
            track_events["EventDriverSampleCount"].sum()
        ),
        "TrackHistoryReliability": min(race_count / 3, 1.0),
        "TrackHistoryDataMissing": False,
    }


def load_available_track_history(
    processed_dir: Path, *, target_season: int, target_round: int
) -> pd.DataFrame:
    """Tam geçmişi tercih ederek yalnız hedef öncesi pist satırlarını okur."""
    processed = Path(processed_dir)
    selected: dict[tuple[int, int], Path] = {}

    sources = (
        (processed / "track_history", r"round_(\d+)_track_rows\.csv", False),
        (processed / "history", r"round_(\d+)_training_rows\.csv", True),
    )
    for root, pattern, replaces_compact in sources:
        if not root.exists():
            continue
        for path in root.glob("*/*.csv"):
            try:
                season = int(path.parent.name)
            except ValueError:
                continue
            match = re.fullmatch(pattern, path.name)
            if match is None:
                continue
            round_number = int(match.group(1))
            before_target = season < target_season or (
                season == target_season and round_number < target_round
            )
            if not before_target:
                continue
            key = (season, round_number)
            if replaces_compact or key not in selected:
                selected[key] = path

    frames = [pd.read_csv(selected[key]) for key in sorted(selected)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_track_features(
    weekend: pd.DataFrame,
    geometry: dict[str, object],
    history: pd.DataFrame,
    *,
    target_season: int,
    target_round: int,
) -> pd.DataFrame:
    """Tek yarışlık pist özelliklerini tüm pilot satırlarına ekler."""
    missing = sorted(set(TRACK_GEOMETRY_COLUMNS).difference(geometry))
    if missing:
        raise TrackDataError(
            f"Pist geometrisinde eksik alanlar var: {', '.join(missing)}"
        )
    collisions = sorted(
        set(weekend.columns).intersection(
            set(TRACK_GEOMETRY_COLUMNS) | set(TRACK_HISTORY_COLUMNS)
        )
    )
    if collisions:
        raise TrackDataError(
            f"Pist özellikleri mevcut tabloyla çakışıyor: {', '.join(collisions)}"
        )
    profile = build_track_history_features(
        history,
        track_id=str(geometry["TrackId"]),
        target_season=target_season,
        target_round=target_round,
    )
    result = weekend.copy()
    for column, value in {**geometry, **profile}.items():
        result[column] = value
    return result
