"""Pist geometrisi ve geçmiş pozisyon değişimi özellikleri."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests


MULTIVIEWER_CIRCUIT_URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"


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


def fetch_multiviewer_circuit(
    track_id: str,
    year: int,
    *,
    get: Callable[..., requests.Response] = requests.get,
    timeout: float = 30.0,
) -> dict[str, object] | None:
    """Pist çizimini getirir; ulaşılamazsa geometri özellikleri eksik kalır."""
    try:
        response = get(
            MULTIVIEWER_CIRCUIT_URL.format(key=track_id, year=year),
            headers={"User-Agent": "f1-prediction-v2/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        circuit = response.json()
    except (requests.RequestException, ValueError):
        return None
    return circuit if isinstance(circuit, dict) else None


def _circuit_corners(circuit: dict[str, object]) -> pd.DataFrame:
    corners = pd.DataFrame(circuit.get("corners") or [])
    return corners.rename(columns={"number": "Number", "letter": "Letter"})


def _circuit_length_km(circuit: dict[str, object]) -> float | None:
    """Pist çizgisinin çevresini hesaplar; koordinatlar metrenin onda biridir."""
    x = pd.to_numeric(pd.Series(circuit.get("x") or [], dtype=float), errors="coerce")
    y = pd.to_numeric(pd.Series(circuit.get("y") or [], dtype=float), errors="coerce")
    if len(x) < 2 or len(x) != len(y) or x.isna().any() or y.isna().any():
        return None
    closed_x = np.append(x.to_numpy(), x.iloc[0])
    closed_y = np.append(y.to_numpy(), y.iloc[0])
    length_km = float(np.hypot(np.diff(closed_x), np.diff(closed_y)).sum()) / 10_000
    if not 1.0 <= length_km <= 10.0:
        return None
    return length_km


def track_identity_from_history(
    history: pd.DataFrame, event_name: str
) -> tuple[str, str] | None:
    """Aynı etkinliğin önceki kaydından pist kimliğini bulur."""
    required = {"EventName", "TrackId", "TrackName"}
    if history.empty or not required.issubset(history.columns):
        return None
    rows = history[history["EventName"].astype(str) == event_name].dropna(
        subset=["TrackId", "TrackName"]
    )
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    return str(latest["TrackId"]), str(latest["TrackName"])


def build_track_geometry_features(
    track_id: str,
    track_name: str,
    *,
    year: int,
    circuit_getter: Callable[..., dict[str, object] | None] = fetch_multiviewer_circuit,
) -> dict[str, object]:
    """Pist kimliği ve MultiViewer çiziminden yaklaşık mesafe ve viraj sayısı üretir."""
    circuit = circuit_getter(track_id, year)
    corner_count = _corner_count(_circuit_corners(circuit)) if circuit else None
    length_km = _circuit_length_km(circuit) if circuit else None
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
        "TrackGeometrySource": "OPENF1_AND_MULTIVIEWER",
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
