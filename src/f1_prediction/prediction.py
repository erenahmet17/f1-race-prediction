"""Gerçek yarış haftası için sızıntısız tahmin akışı."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backfill import backfill_training_history
from .collection import (
    CollectedWeekend,
    WeekendDataPaths,
    collect_pre_race_weekend,
)
from .dashboard import build_race_view
from .history import collect_completed_round
from .history import load_history_before_round
from .modeling import (
    BacktestResult,
    UpcomingPredictionResult,
    predict_upcoming_round,
    walk_forward_backtest,
)


class PreRacePredictionError(RuntimeError):
    """Yarış öncesi tahmin güvenle kaydedilemediğinde oluşur."""


@dataclass(frozen=True)
class PreRacePredictionRun:
    weekend: CollectedWeekend
    model: UpcomingPredictionResult
    calibration: BacktestResult | None
    prediction_path: Path | None


@dataclass(frozen=True)
class RetrospectivePredictionRun:
    weekend: CollectedWeekend
    model: UpcomingPredictionResult
    prediction_path: Path
    mae: float
    clean_mae: float | None
    evaluated_rows: pd.DataFrame


def _prediction_path(
    paths: WeekendDataPaths,
    season: int,
    round_number: int,
) -> Path:
    return (
        paths.processed
        / "predictions"
        / str(season)
        / f"round_{round_number:02d}_prediction.csv"
    )


def _retrospective_path(
    paths: WeekendDataPaths, season: int, round_number: int
) -> Path:
    return (
        paths.processed
        / "retrospectives"
        / str(season)
        / f"round_{round_number:02d}_retrospective.csv"
    )


def _write_csv_once(predictions: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PreRacePredictionError(f"Mevcut tahmin dosyası korunuyor: {destination}")
    temporary = destination.with_suffix(".csv.tmp")
    try:
        predictions.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_prediction(
    predictions: pd.DataFrame,
    paths: WeekendDataPaths,
    season: int,
    round_number: int,
) -> Path:
    return _write_csv_once(
        predictions, _prediction_path(paths, season, round_number)
    )


def _predict_with_prior_history(
    history: pd.DataFrame,
    weekend: CollectedWeekend,
    *,
    min_train_rounds: int,
) -> tuple[UpcomingPredictionResult, BacktestResult | None]:
    completed_round_count = int(history["RoundNumber"].nunique())
    calibration: BacktestResult | None = None
    calibration_predictions = pd.DataFrame()
    if completed_round_count > min_train_rounds:
        calibration = walk_forward_backtest(
            history,
            min_train_rounds=min_train_rounds,
        )
        calibration_predictions = calibration.predictions

    model = predict_upcoming_round(
        history,
        weekend.features,
        calibration_predictions=calibration_predictions,
        min_train_rounds=min_train_rounds,
    )
    return model, calibration


def create_pre_race_prediction(
    season: int,
    round_number: int,
    *,
    paths: WeekendDataPaths,
    save: bool = True,
    min_train_rounds: int = 1,
) -> PreRacePredictionRun:
    """Final grid açıklandıktan sonra veriyi toplar, modeli eğitir ve tahmini kaydeder."""
    destination = _prediction_path(paths, season, round_number)
    if save and destination.exists():
        raise PreRacePredictionError(
            f"Bu yarışın tahmini zaten kayıtlı; mevcut dosya korunuyor: {destination}"
        )
    history = load_history_before_round(
        paths,
        season=season,
        prediction_round=round_number,
    )
    weekend = collect_pre_race_weekend(
        season,
        round_number,
        paths=paths,
        save=save,
    )
    race_start_utc = weekend.race_start_utc
    if race_start_utc is None or pd.isna(race_start_utc):
        raise PreRacePredictionError(
            "Yarışın UTC başlangıç saati doğrulanamadı; yarış öncesi tahmin durduruldu"
        )
    if pd.Timestamp.now(tz="UTC") >= race_start_utc:
        raise PreRacePredictionError(
            "Yarış başlamış; yeni yarış öncesi tahmin kaydedilemez"
        )
    model, calibration = _predict_with_prior_history(
        history, weekend, min_train_rounds=min_train_rounds,
    )
    if save and destination.exists():
        raise PreRacePredictionError(
            f"Bu yarışın tahmini işlem sırasında kaydedilmiş: {destination}"
        )
    prediction_path = (
        _write_prediction(model.predictions, paths, season, round_number)
        if save
        else None
    )
    return PreRacePredictionRun(
        weekend=weekend,
        model=model,
        calibration=calibration,
        prediction_path=prediction_path,
    )


def create_retrospective_prediction(
    season: int,
    round_number: int,
    *,
    paths: WeekendDataPaths,
    min_train_rounds: int = 1,
) -> RetrospectivePredictionRun:
    """Yarış bitince, yalnız yarış öncesi girdilerle ayrı etiketli deneme yapar."""
    destination = _retrospective_path(paths, season, round_number)
    if _prediction_path(paths, season, round_number).exists():
        raise PreRacePredictionError(
            "Bu round için gerçek yarış öncesi tahmin mevcut; geriye dönük deneme yapılmaz"
        )
    if destination.exists():
        raise PreRacePredictionError(
            f"Bu round için geriye dönük deneme zaten kayıtlı: {destination}"
        )
    if round_number <= 1:
        raise PreRacePredictionError("Sezonun ilk yarışı için tahmin üretilmez")

    # Yalnız 1..N-1 geçmişi hazırlanır; hedef yarış eğitime hiçbir zaman girmez.
    backfill_training_history(season, round_number - 1, paths=paths)
    history = load_history_before_round(
        paths, season=season, prediction_round=round_number
    )
    weekend = collect_pre_race_weekend(
        season, round_number, paths=paths, save=False
    )
    race_start = weekend.race_start_utc
    if race_start is None or pd.isna(race_start):
        raise PreRacePredictionError("Yarışın UTC başlangıç saati doğrulanamadı")
    race_start = pd.Timestamp(race_start)
    if race_start.tzinfo is None:
        raise PreRacePredictionError("Yarış başlangıç saati zaman dilimi içermiyor")
    if pd.Timestamp.now(tz="UTC") < race_start:
        raise PreRacePredictionError(
            "Yarış henüz başlamadı; canlı yarış öncesi tahmin bölümünü kullanın"
        )
    grid_published = pd.Timestamp(weekend.final_grid.document.published_at)
    if grid_published.tzinfo is None or grid_published >= race_start:
        raise PreRacePredictionError(
            "FIA final gridinin yarıştan önce yayımlandığı doğrulanamadı"
        )

    model, _ = _predict_with_prior_history(
        history, weekend, min_train_rounds=min_train_rounds
    )
    target_path = (
        paths.processed / "history" / str(season)
        / f"round_{round_number:02d}_training_rows.csv"
    )
    if target_path.exists():
        actual = pd.read_csv(target_path)
    else:
        completed = collect_completed_round(
            season, round_number, paths=paths, save=True, pre_race=weekend
        )
        actual = completed.rows
    required_actual = {"Season", "RoundNumber", "DriverId", "ActualPosition", "CleanMAEEligible"}
    if actual.empty or not required_actual.issubset(actual.columns):
        raise PreRacePredictionError("Kaydedilmiş yarış sonucu eksik veya geçersiz")
    if set(pd.to_numeric(actual["Season"], errors="coerce").dropna()) != {season} or set(
        pd.to_numeric(actual["RoundNumber"], errors="coerce").dropna()
    ) != {round_number}:
        raise PreRacePredictionError("Yarış sonucu sezon/round ile uyuşmuyor")
    predicted_ids = model.predictions["DriverId"]
    actual_ids = actual["DriverId"]
    if (
        predicted_ids.isna().any()
        or actual_ids.isna().any()
        or predicted_ids.duplicated().any()
        or actual_ids.duplicated().any()
        or set(predicted_ids) != set(actual_ids)
        or pd.to_numeric(actual["ActualPosition"], errors="coerce").isna().any()
    ):
        raise PreRacePredictionError("Tahmin ile yarış sonucu pilot listesi uyuşmuyor")
    view = build_race_view(
        saved_prediction=pd.DataFrame(),
        retrospective_prediction=model.predictions,
        backtest_prediction=pd.DataFrame(),
        history=actual,
    )
    if view.mae is None:
        raise PreRacePredictionError("Yarış sonucu için MAE hesaplanamadı")
    _write_csv_once(model.predictions, destination)
    return RetrospectivePredictionRun(
        weekend=weekend,
        model=model,
        prediction_path=destination,
        mae=view.mae,
        clean_mae=view.clean_mae,
        evaluated_rows=view.rows,
    )
