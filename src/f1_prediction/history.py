"""Tamamlanan yarışları saklar ve sızıntısız sezon geçmişini seçer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fastf1
import pandas as pd

from .collection import (
    CollectedWeekend,
    WeekendCollectionError,
    WeekendDataPaths,
    collect_pre_race_weekend,
)
from .fia_race import FiaRaceSnapshot, fetch_fia_final_race_classification
from .identity import normalize_fastf1_driver_ids
from .race import RACE_TARGET_COLUMNS, build_race_targets


class HistoryDataError(ValueError):
    """Geçmiş yarış dosyaları eksik veya tutarsızsa oluşur."""


class InsufficientHistoryError(HistoryDataError):
    """Tahmin roundundan önce yeterli tamamlanmış yarış yoksa oluşur."""


@dataclass(frozen=True)
class CompletedRound:
    rows: pd.DataFrame
    pre_race: CollectedWeekend
    final_race: FiaRaceSnapshot
    history_path: Path | None


def build_completed_round_rows(
    pre_race_features: pd.DataFrame,
    race_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Yarış öncesi özelliklere yarıştan sonra yalnız hedef sütunlarını ekler."""
    pre_required = {"Season", "RoundNumber", "DriverId", "Driver", "Team", "StartOrder"}
    target_required = {"DriverId", "RaceDriver", "RaceTeam", *RACE_TARGET_COLUMNS}
    missing_pre = sorted(pre_required.difference(pre_race_features.columns))
    missing_target = sorted(target_required.difference(race_targets.columns))
    if missing_pre:
        raise HistoryDataError(
            f"Yarış öncesi tabloda eksik sütunlar var: {', '.join(missing_pre)}"
        )
    if missing_target:
        raise HistoryDataError(
            f"Yarış hedeflerinde eksik sütunlar var: {', '.join(missing_target)}"
        )
    if pre_race_features["DriverId"].duplicated().any():
        raise HistoryDataError("Yarış öncesi tabloda tekrarlanan pilot var")
    if race_targets["DriverId"].duplicated().any():
        raise HistoryDataError("Yarış hedeflerinde tekrarlanan pilot var")

    pre_ids = set(pre_race_features["DriverId"])
    target_ids = set(race_targets["DriverId"])
    if pre_ids != target_ids:
        raise HistoryDataError(
            "Yarış öncesi ve sonrası pilot listeleri uyuşmuyor. "
            f"Eksik: {sorted(pre_ids - target_ids)}; Fazla: {sorted(target_ids - pre_ids)}"
        )

    target_columns = ["DriverId", "RaceDriver", "RaceTeam", *RACE_TARGET_COLUMNS]
    combined = pre_race_features.merge(
        race_targets[target_columns],
        on="DriverId",
        how="inner",
        validate="one_to_one",
    )
    driver_mismatch = combined["Driver"].astype(str) != combined["RaceDriver"].astype(str)
    team_mismatch = combined["Team"].astype(str) != combined["RaceTeam"].astype(str)
    if driver_mismatch.any() or team_mismatch.any():
        bad = sorted(combined.loc[driver_mismatch | team_mismatch, "DriverId"].astype(str))
        raise HistoryDataError(
            f"Yarış sonucu kimlikleri yarış öncesi tabloyla uyuşmuyor: {', '.join(bad)}"
        )
    return combined.drop(columns=["RaceDriver", "RaceTeam"]).sort_values(
        "StartOrder"
    ).reset_index(drop=True)


def _history_path(
    paths: WeekendDataPaths, season: int, round_number: int
) -> Path:
    return (
        paths.processed
        / "history"
        / str(season)
        / f"round_{round_number:02d}_training_rows.csv"
    )


def _write_history(
    rows: pd.DataFrame,
    paths: WeekendDataPaths,
    season: int,
    round_number: int,
) -> Path:
    destination = _history_path(paths, season, round_number)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    try:
        rows.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def collect_completed_round(
    season: int,
    round_number: int,
    *,
    paths: WeekendDataPaths,
    save: bool = True,
    event_getter: Callable[[int, int], object] = fastf1.get_event,
    cache_enabler: Callable[[str], None] = fastf1.Cache.enable_cache,
    pre_race: CollectedWeekend | None = None,
) -> CompletedRound:
    """Yarış bittikten sonra pre-race özelliklerini resmî sonuçla eşler."""
    if pre_race is None:
        pre_race = collect_pre_race_weekend(
            season,
            round_number,
            paths=paths,
            save=save,
            event_getter=event_getter,
            cache_enabler=cache_enabler,
        )
    else:
        seasons = set(pd.to_numeric(pre_race.features["Season"], errors="coerce").dropna())
        rounds = set(pd.to_numeric(pre_race.features["RoundNumber"], errors="coerce").dropna())
        if seasons != {season} or rounds != {round_number}:
            raise HistoryDataError("Verilen yarış öncesi tablo sezon/round ile uyuşmuyor")
    try:
        event = event_getter(season, round_number)
        race_session = event.get_session("Race")
        race_session.load(
            laps=False,
            telemetry=False,
            weather=False,
            messages=False,
        )
    except Exception as exc:
        raise WeekendCollectionError(f"Race sonucu yüklenemedi: {exc}") from exc

    race_results = normalize_fastf1_driver_ids(race_session.results.copy())
    final_race = fetch_fia_final_race_classification(
        pre_race.event_name,
        season,
        race_results,
        raw_data_dir=paths.raw,
    )
    race_targets = build_race_targets(
        race_results,
        pre_race.final_grid.features,
        final_race.classification,
    )
    rows = build_completed_round_rows(pre_race.features, race_targets)
    history_path = (
        _write_history(rows, paths, season, round_number) if save else None
    )
    return CompletedRound(
        rows=rows,
        pre_race=pre_race,
        final_race=final_race,
        history_path=history_path,
    )


def select_history_before_round(
    history: pd.DataFrame,
    *,
    season: int,
    prediction_round: int,
) -> pd.DataFrame:
    """Aynı sezondan yalnız tahmin roundundan önceki satırları döndürür."""
    if prediction_round <= 1:
        raise InsufficientHistoryError("Sezonun ilk yarışı için tahmin üretilmez")
    required = {"Season", "RoundNumber", "DriverId", "ActualPosition"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise HistoryDataError(f"Geçmiş tabloda eksik sütunlar var: {', '.join(missing)}")
    selected = history[
        (pd.to_numeric(history["Season"], errors="coerce") == season)
        & (pd.to_numeric(history["RoundNumber"], errors="coerce") < prediction_round)
    ].copy()
    if selected.empty:
        raise InsufficientHistoryError(
            f"{season} round {prediction_round} öncesinde tamamlanmış yarış yok"
        )
    return selected.sort_values(["RoundNumber", "ActualPosition"]).reset_index(drop=True)


def load_history_before_round(
    paths: WeekendDataPaths,
    *,
    season: int,
    prediction_round: int,
) -> pd.DataFrame:
    """Diskteki eksiksiz round 1..N-1 geçmişini sızıntısız biçimde yükler."""
    if prediction_round <= 1:
        raise InsufficientHistoryError("Sezonun ilk yarışı için tahmin üretilmez")

    directory = paths.processed / "history" / str(season)
    files_by_round: dict[int, Path] = {}
    pattern = re.compile(r"round_(\d+)_training_rows\.csv$")
    if directory.exists():
        for path in directory.glob("round_*_training_rows.csv"):
            match = pattern.fullmatch(path.name)
            if match:
                round_number = int(match.group(1))
                if round_number in files_by_round:
                    raise HistoryDataError(
                        f"Round {round_number} için birden fazla geçmiş dosyası var"
                    )
                files_by_round[round_number] = path

    required_rounds = set(range(1, prediction_round))
    missing_rounds = sorted(required_rounds.difference(files_by_round))
    if missing_rounds:
        raise InsufficientHistoryError(
            "Tahmin için önce tamamlanması gereken roundlar eksik: "
            + ", ".join(map(str, missing_rounds))
        )

    frames: list[pd.DataFrame] = []
    for round_number in sorted(required_rounds):
        frame = pd.read_csv(files_by_round[round_number])
        if frame.empty:
            raise HistoryDataError(f"Round {round_number} geçmiş dosyası boş")
        seasons = set(pd.to_numeric(frame.get("Season"), errors="coerce").dropna())
        rounds = set(pd.to_numeric(frame.get("RoundNumber"), errors="coerce").dropna())
        if seasons != {season} or rounds != {round_number}:
            raise HistoryDataError(
                f"Round {round_number} dosyasının sezon/round kimliği hatalı"
            )
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    return select_history_before_round(
        history,
        season=season,
        prediction_round=prediction_round,
    )
