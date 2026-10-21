"""FastF1 ve FIA kaynaklarından yarış öncesi özellik tablosunu üretir."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fastf1
import pandas as pd

from .fia_grid import FiaGridSnapshot, fetch_fia_final_grid
from .identity import normalize_fastf1_driver_ids
from .practice import PracticeFeatures, build_practice_features
from .qualifying import build_qualifying_features
from .sprint import build_sprint_features
from .track import (
    attach_track_features,
    build_track_geometry_features,
    load_available_track_history,
)
from .weekend import build_weekend_feature_table


class WeekendCollectionError(RuntimeError):
    """Bir yarış haftasının kaynak verileri toplanamadığında oluşur."""


@dataclass(frozen=True)
class WeekendDataPaths:
    """Yarış haftası verilerinin proje içindeki konumları."""

    cache: Path
    raw: Path
    processed: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "WeekendDataPaths":
        data = Path(project_root).resolve() / "data"
        return cls(
            cache=data / "cache",
            raw=data / "raw",
            processed=data / "processed",
        )

    def ensure(self) -> None:
        for directory in (self.cache, self.raw, self.processed):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EventSessions:
    practice: tuple[str, ...]
    qualifying: str
    sprint: str | None

    @property
    def is_sprint_weekend(self) -> bool:
        return self.sprint is not None


@dataclass(frozen=True)
class CollectedWeekend:
    features: pd.DataFrame
    event_name: str
    sessions: EventSessions
    final_grid: FiaGridSnapshot
    feature_path: Path | None
    race_start_utc: pd.Timestamp | None = None


def classify_event_sessions(event: object) -> EventSessions:
    """Takvim satırından gerekli seansları seçer; Sprint Qualifying'i yok sayar."""
    names = tuple(
        str(name).strip()
        for index in range(1, 6)
        if (name := event.get(f"Session{index}")) is not None
        and str(name).strip()
    )
    practice = tuple(name for name in names if name.lower().startswith("practice"))
    qualifying = next(
        (name for name in names if name.lower() == "qualifying"), None
    )
    sprint = next((name for name in names if name.lower() == "sprint"), None)

    if not practice:
        raise WeekendCollectionError("Takvimde practice seansı bulunamadı")
    if qualifying is None:
        raise WeekendCollectionError("Takvimde ana Qualifying seansı bulunamadı")
    return EventSessions(
        practice=practice,
        qualifying=qualifying,
        sprint=sprint,
    )


def _practice_code(session_name: str, fallback_number: int) -> str:
    match = re.search(r"(\d+)$", session_name)
    number = int(match.group(1)) if match else fallback_number
    return f"FP{number}"


def _load_session(
    event: object, session_name: str, *, telemetry: bool = False
) -> object:
    try:
        session = event.get_session(session_name)
        session.load(
            laps=True,
            telemetry=telemetry,
            weather=False,
            messages=False,
        )
        return session
    except Exception as exc:
        raise WeekendCollectionError(
            f"{session_name} seansı yüklenemedi: {exc}"
        ) from exc


def _align_team_names(
    frame: pd.DataFrame,
    qualifying_results: pd.DataFrame,
    *,
    driver_column: str,
    team_column: str,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    """Practice/sprint takım adlarını ana qualifying kadrosuyla aynılaştırır."""
    required_frame = {driver_column, team_column}
    required_qualifying = {"Abbreviation", "TeamName"}
    if not required_frame.issubset(frame.columns):
        return frame.copy()
    missing_qualifying = sorted(required_qualifying.difference(qualifying_results.columns))
    if missing_qualifying:
        raise WeekendCollectionError(
            "Qualifying takım eşlemesinde eksik sütunlar var: "
            + ", ".join(missing_qualifying)
        )

    mapping = qualifying_results[["Abbreviation", "TeamName"]].copy()
    mapping["Abbreviation"] = mapping["Abbreviation"].astype(str).str.strip().str.upper()
    mapping["TeamName"] = mapping["TeamName"].astype("string").str.strip()
    if mapping["Abbreviation"].duplicated().any():
        raise WeekendCollectionError("Qualifying takım eşlemesinde pilot tekrarı var")
    if mapping["TeamName"].isna().any() or mapping["TeamName"].eq("").any():
        raise WeekendCollectionError("Qualifying takım eşlemesinde boş takım adı var")
    team_by_driver = mapping.set_index("Abbreviation")["TeamName"]

    aligned = frame.copy()
    driver_codes = aligned[driver_column].astype(str).str.strip().str.upper()
    canonical_teams = driver_codes.map(team_by_driver)
    aligned.loc[canonical_teams.notna(), team_column] = canonical_teams.dropna()
    if drop_unmatched:
        aligned = aligned.loc[canonical_teams.notna()].copy()
    return aligned


def _write_features(
    features: pd.DataFrame,
    paths: WeekendDataPaths,
    season: int,
    round_number: int,
) -> Path:
    directory = paths.processed / "pre_race" / str(season)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"round_{round_number:02d}_weekend_features.csv"
    temporary = destination.with_suffix(".csv.tmp")
    try:
        features.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def collect_pre_race_weekend(
    season: int,
    round_number: int,
    *,
    paths: WeekendDataPaths,
    save: bool = True,
    event_getter: Callable[[int, int], object] = fastf1.get_event,
    cache_enabler: Callable[[str], None] = fastf1.Cache.enable_cache,
) -> CollectedWeekend:
    """Practice, sprint, qualifying ve resmî gridden yarış öncesi tablo kurar."""
    if season < 1950:
        raise ValueError("season geçerli bir F1 sezonu olmalı")
    if round_number < 1:
        raise ValueError("round_number en az 1 olmalı")

    paths.ensure()
    cache_enabler(str(paths.cache))
    try:
        event = event_getter(season, round_number)
    except Exception as exc:
        raise WeekendCollectionError(
            f"{season} sezonu {round_number}. round takvimde bulunamadı: {exc}"
        ) from exc

    event_name = str(event.get("EventName", "")).strip()
    event_round = pd.to_numeric(event.get("RoundNumber"), errors="coerce")
    if not event_name:
        raise WeekendCollectionError("Takvimde EventName eksik")
    if pd.isna(event_round) or int(event_round) != round_number:
        raise WeekendCollectionError("Takvim round numarası istekle uyuşmuyor")

    sessions = classify_event_sessions(event)
    race_start_utc: pd.Timestamp | None = None
    for session_index in range(1, 6):
        if str(event.get(f"Session{session_index}", "")).strip().lower() == "race":
            race_date = pd.to_datetime(
                event.get(f"Session{session_index}DateUtc"),
                utc=True,
                errors="coerce",
            )
            if not pd.isna(race_date):
                race_start_utc = race_date
            break
    practice_laps: dict[str, pd.DataFrame] = {}
    for index, session_name in enumerate(sessions.practice, start=1):
        session = _load_session(event, session_name)
        practice_laps[_practice_code(session_name, index)] = session.laps.copy()
    sprint_session: object | None = None
    if sessions.sprint is not None:
        sprint_session = _load_session(event, sessions.sprint)

    qualifying_session = _load_session(
        event, sessions.qualifying, telemetry=True
    )
    qualifying_results = normalize_fastf1_driver_ids(
        qualifying_session.results.copy()
    )
    qualifying_features = build_qualifying_features(qualifying_results)
    practice_laps = {
        code: _align_team_names(
            laps,
            qualifying_results,
            driver_column="Driver",
            team_column="Team",
            drop_unmatched=True,
        )
        for code, laps in practice_laps.items()
    }
    practice_features: PracticeFeatures = build_practice_features(
        practice_laps,
        expected_sessions=len(sessions.practice),
    )

    sprint_features: pd.DataFrame | None = None
    if sprint_session is not None:
        sprint_features = build_sprint_features(
            _align_team_names(
                sprint_session.laps.copy(),
                qualifying_results,
                driver_column="Driver",
                team_column="Team",
                drop_unmatched=True,
            ),
            _align_team_names(
                sprint_session.results.copy(),
                qualifying_results,
                driver_column="Abbreviation",
                team_column="TeamName",
                drop_unmatched=True,
            ),
        )
    final_grid = fetch_fia_final_grid(
        event_name,
        season,
        qualifying_results,
        raw_data_dir=paths.raw,
    )

    features = build_weekend_feature_table(
        qualifying_features,
        final_grid.features,
        season=season,
        round_number=round_number,
        event_name=event_name,
        practice_drivers=practice_features.drivers,
        practice_teams=practice_features.teams,
        sprint=sprint_features,
        is_sprint_weekend=sessions.is_sprint_weekend,
    )
    geometry = build_track_geometry_features(qualifying_session)
    track_history = load_available_track_history(
        paths.processed,
        target_season=season,
        target_round=round_number,
    )
    features = attach_track_features(
        features,
        geometry,
        track_history,
        target_season=season,
        target_round=round_number,
    )
    feature_path = (
        _write_features(features, paths, season, round_number) if save else None
    )
    return CollectedWeekend(
        features=features,
        event_name=event_name,
        sessions=sessions,
        final_grid=final_grid,
        feature_path=feature_path,
        race_start_utc=race_start_utc,
    )
