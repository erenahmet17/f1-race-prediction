"""Geçmiş pist verisini ve güncel sezon eğitim satırlarını topluca üretir."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fastf1
import pandas as pd
import requests

from .collection import WeekendDataPaths
from .history import CompletedRound, collect_completed_round
from .track import _session_circuit_identity


class BackfillError(RuntimeError):
    """Toplu veri toplama güvenle devam edemediğinde oluşur."""


TRACK_ROW_REQUIRED_COLUMNS = {
    "Season",
    "RoundNumber",
    "EventName",
    "TrackId",
    "TrackName",
    "DriverId",
    "StartOrder",
    "ActualPosition",
    "RaceStatus",
    "CleanMAEEligible",
    "TrackHistorySource",
}
TRAINING_ROW_REQUIRED_COLUMNS = {
    "Season",
    "RoundNumber",
    "DriverId",
    "TrackId",
    "StartOrder",
    "ActualPosition",
    "CleanMAEEligible",
}


@dataclass(frozen=True)
class BackfillSummary:
    collected_rounds: tuple[int, ...]
    skipped_rounds: tuple[int, ...]
    output_paths: tuple[Path, ...]


def _completed_status(status: object) -> bool:
    value = str(status).strip().lower()
    return value in {"finished", "lapped"} or bool(
        re.search(r"\+\s*\d+\s*lap", value)
    )


def build_compact_track_history_rows(
    results: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    event_name: str,
    track_id: str,
    track_name: str,
    source: str = "FASTF1_RACE_RESULTS",
) -> pd.DataFrame:
    """Eski yarıştan yalnız pist hareketi için gereken sütunları çıkarır."""
    required = {"DriverId", "GridPosition", "Position", "Status"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise BackfillError(
            f"FastF1 yarış sonucunda eksik sütunlar var: {', '.join(missing)}"
        )
    if results.empty:
        raise BackfillError("FastF1 yarış sonucu boş")
    if results["DriverId"].isna().any() or results["DriverId"].duplicated().any():
        raise BackfillError("FastF1 yarış sonucu pilot kimlikleri geçersiz")

    rows = results[["DriverId", "GridPosition", "Position", "Status"]].copy()
    rows = rows.rename(
        columns={
            "GridPosition": "StartOrder",
            "Position": "ActualPosition",
            "Status": "RaceStatus",
        }
    )
    rows["StartOrder"] = pd.to_numeric(rows["StartOrder"], errors="coerce")
    rows["ActualPosition"] = pd.to_numeric(
        rows["ActualPosition"], errors="coerce"
    )
    valid_grid = rows["StartOrder"].notna() & (rows["StartOrder"] > 0)
    valid_finish = rows["ActualPosition"].notna() & (rows["ActualPosition"] > 0)
    rows.loc[~valid_grid, "StartOrder"] = pd.NA
    rows["CleanMAEEligible"] = (
        rows["RaceStatus"].map(_completed_status) & valid_grid & valid_finish
    )
    rows.insert(0, "TrackHistorySource", str(source))
    rows.insert(0, "TrackName", str(track_name))
    rows.insert(0, "TrackId", str(track_id))
    rows.insert(0, "EventName", str(event_name))
    rows.insert(0, "RoundNumber", int(round_number))
    rows.insert(0, "Season", int(season))
    return rows.reset_index(drop=True)


def fetch_jolpica_race_results(
    season: int,
    round_number: int,
    *,
    requester: Callable[..., object] = requests.get,
) -> pd.DataFrame:
    """FastF1 sonuç tablosu eksikse Jolpica'nın Ergast uyumlu API'sini kullanır."""
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_number}/results.json"
    try:
        response = requester(url, timeout=30)
        response.raise_for_status()
        races = response.json()["MRData"]["RaceTable"]["Races"]
        results = races[0]["Results"] if races else []
    except Exception as exc:
        raise BackfillError(
            f"Jolpica yarış sonucu yüklenemedi: {season} round {round_number}"
        ) from exc
    if not results:
        raise BackfillError(
            f"Jolpica yarış sonucu boş: {season} round {round_number}"
        )
    return pd.DataFrame(
        {
            "DriverId": [item["Driver"]["driverId"] for item in results],
            "GridPosition": [item.get("grid") for item in results],
            "Position": [item.get("position") for item in results],
            "Status": [item.get("status") for item in results],
        }
    )


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _track_path(
    paths: WeekendDataPaths, season: int, round_number: int
) -> Path:
    return (
        paths.processed
        / "track_history"
        / str(season)
        / f"round_{round_number:02d}_track_rows.csv"
    )


def _training_path(
    paths: WeekendDataPaths, season: int, round_number: int
) -> Path:
    return (
        paths.processed
        / "history"
        / str(season)
        / f"round_{round_number:02d}_training_rows.csv"
    )


def _validate_saved_rows(
    path: Path,
    *,
    season: int,
    round_number: int,
    required_columns: set[str],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise BackfillError(f"Mevcut dosya okunamadı: {path}") from exc
    missing = sorted(required_columns.difference(frame.columns))
    if frame.empty or missing:
        detail = f"; eksik sütunlar: {', '.join(missing)}" if missing else ""
        raise BackfillError(f"Mevcut dosya geçersiz veya boş: {path}{detail}")
    seasons = set(pd.to_numeric(frame["Season"], errors="coerce").dropna())
    rounds = set(pd.to_numeric(frame["RoundNumber"], errors="coerce").dropna())
    if seasons != {season} or rounds != {round_number}:
        raise BackfillError(f"Mevcut dosyanın sezon/round kimliği hatalı: {path}")
    if frame["DriverId"].isna().any() or frame["DriverId"].duplicated().any():
        raise BackfillError(f"Mevcut dosyanın pilot kimlikleri hatalı: {path}")
    return frame


def collect_compact_track_round(
    season: int,
    round_number: int,
    *,
    paths: WeekendDataPaths,
    session_getter: Callable[..., object] = fastf1.get_session,
    cache_enabler: Callable[[str], None] = fastf1.Cache.enable_cache,
    fallback_results_getter: Callable[[int, int], pd.DataFrame] = (
        fetch_jolpica_race_results
    ),
) -> Path:
    """Tek eski yarışı pist karakteri için kompakt biçimde kaydeder."""
    paths.ensure()
    cache_enabler(str(paths.cache))
    try:
        session = session_getter(season, round_number, "R")
        session.load(
            laps=False,
            telemetry=False,
            weather=False,
            messages=False,
        )
        track_id, track_name = _session_circuit_identity(session)
        event_name = str(session.event["EventName"])
    except Exception as exc:
        raise BackfillError(
            f"{season} round {round_number} yarış sonucu yüklenemedi: {exc}"
        ) from exc

    row_context = {
        "season": season,
        "round_number": round_number,
        "event_name": event_name,
        "track_id": track_id,
        "track_name": track_name,
    }
    try:
        rows = build_compact_track_history_rows(
            session.results.copy(),
            **row_context,
        )
    except BackfillError:
        fallback_results = fallback_results_getter(season, round_number)
        rows = build_compact_track_history_rows(
            fallback_results,
            source="JOLPICA_RESULTS_FALLBACK",
            **row_context,
        )
    return _atomic_csv(rows, _track_path(paths, season, round_number))


def _schedule_rounds(
    season: int,
    *,
    schedule_getter: Callable[..., pd.DataFrame],
    through_round: int | None,
) -> list[int]:
    try:
        schedule = schedule_getter(season, include_testing=False)
    except Exception as exc:
        raise BackfillError(f"{season} takvimi yüklenemedi: {exc}") from exc
    if "RoundNumber" not in schedule.columns:
        raise BackfillError(f"{season} takviminde RoundNumber yok")
    rounds = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                schedule["RoundNumber"], errors="coerce"
            ).dropna()
            if int(value) > 0
        }
    )
    if through_round is not None:
        rounds = [round_number for round_number in rounds if round_number <= through_round]
    if not rounds:
        raise BackfillError(f"{season} için işlenecek yarış bulunamadı")
    return rounds


def backfill_track_history(
    start_season: int,
    end_season: int,
    *,
    paths: WeekendDataPaths,
    through_round: int | None = None,
    force: bool = False,
    schedule_getter: Callable[..., pd.DataFrame] = fastf1.get_event_schedule,
    collector: Callable[..., Path] = collect_compact_track_round,
) -> dict[int, BackfillSummary]:
    """Sezon aralığındaki yarışları yalnız pist karakteri için toplar."""
    if start_season > end_season:
        raise ValueError("start_season end_season değerinden büyük olamaz")
    if through_round is not None and start_season != end_season:
        raise ValueError("through_round yalnız tek sezon işlenirken kullanılabilir")

    summaries: dict[int, BackfillSummary] = {}
    for season in range(start_season, end_season + 1):
        rounds = _schedule_rounds(
            season,
            schedule_getter=schedule_getter,
            through_round=through_round,
        )
        collected: list[int] = []
        skipped: list[int] = []
        outputs: list[Path] = []
        for round_number in rounds:
            destination = _track_path(paths, season, round_number)
            if destination.exists() and not force:
                _validate_saved_rows(
                    destination,
                    season=season,
                    round_number=round_number,
                    required_columns=TRACK_ROW_REQUIRED_COLUMNS,
                )
                skipped.append(round_number)
                outputs.append(destination)
                continue
            try:
                output = collector(
                    season,
                    round_number,
                    paths=paths,
                )
                _validate_saved_rows(
                    output,
                    season=season,
                    round_number=round_number,
                    required_columns=TRACK_ROW_REQUIRED_COLUMNS,
                )
            except Exception as exc:
                raise BackfillError(
                    f"Pist geçmişi {season} round {round_number} sırasında durdu: {exc}"
                ) from exc
            collected.append(round_number)
            outputs.append(output)
        summaries[season] = BackfillSummary(
            collected_rounds=tuple(collected),
            skipped_rounds=tuple(skipped),
            output_paths=tuple(outputs),
        )
    return summaries


def backfill_training_history(
    season: int,
    through_round: int,
    *,
    paths: WeekendDataPaths,
    force: bool = False,
    collector: Callable[..., CompletedRound] = collect_completed_round,
) -> BackfillSummary:
    """Güncel sezonun FIA doğrulamalı eğitim roundlarını sırayla üretir."""
    if through_round < 1:
        raise ValueError("through_round en az 1 olmalı")
    collected: list[int] = []
    skipped: list[int] = []
    outputs: list[Path] = []
    for round_number in range(1, through_round + 1):
        destination = _training_path(paths, season, round_number)
        if destination.exists() and not force:
            _validate_saved_rows(
                destination,
                season=season,
                round_number=round_number,
                required_columns=TRAINING_ROW_REQUIRED_COLUMNS,
            )
            skipped.append(round_number)
            outputs.append(destination)
            continue
        try:
            result = collector(
                season,
                round_number,
                paths=paths,
                save=True,
            )
            output = result.history_path
            if output is None:
                raise BackfillError("Collector geçmiş dosyası üretmedi")
            _validate_saved_rows(
                output,
                season=season,
                round_number=round_number,
                required_columns=TRAINING_ROW_REQUIRED_COLUMNS,
            )
        except Exception as exc:
            raise BackfillError(
                f"Eğitim geçmişi {season} round {round_number} sırasında durdu: {exc}"
            ) from exc
        collected.append(round_number)
        outputs.append(output)
    return BackfillSummary(
        collected_rounds=tuple(collected),
        skipped_rounds=tuple(skipped),
        output_paths=tuple(outputs),
    )
