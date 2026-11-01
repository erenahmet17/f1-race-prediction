from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import f1_prediction.prediction as prediction
from f1_prediction.collection import WeekendDataPaths
from f1_prediction.modeling import BacktestResult, UpcomingPredictionResult


def test_pre_race_prediction_saves_atomic_output(tmp_path, monkeypatch) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    history = pd.DataFrame(
        {
            "RoundNumber": [1, 2, 3, 4, 5],
            "DriverId": ["A", "A", "A", "A", "A"],
        }
    )
    weekend = SimpleNamespace(
        event_name="Test Grand Prix",
        features=pd.DataFrame({"DriverId": ["A"]}),
        race_start_utc=pd.Timestamp("2099-01-01", tz="UTC"),
    )
    calibration = BacktestResult(
        predictions=pd.DataFrame({"RoundNumber": [5]}),
        round_metrics=pd.DataFrame(),
        summary=pd.DataFrame(),
    )
    model = UpcomingPredictionResult(
        predictions=pd.DataFrame(
            {
                "PredictedPosition": [1],
                "Driver": ["AAA"],
                "TrainingThroughRound": [5],
            }
        ),
        grid_delta_weight=0.5,
        training_rounds=5,
        training_rows=10,
    )
    monkeypatch.setattr(
        prediction,
        "load_history_before_round",
        lambda paths, season, prediction_round: history,
    )
    monkeypatch.setattr(
        prediction,
        "collect_pre_race_weekend",
        lambda season, round_number, paths, save: weekend,
    )
    monkeypatch.setattr(
        prediction,
        "walk_forward_backtest",
        lambda history, min_train_rounds: calibration,
    )
    captured: dict[str, object] = {}

    def fake_predict(history, upcoming, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(prediction, "predict_upcoming_round", fake_predict)

    result = prediction.create_pre_race_prediction(
        2026,
        6,
        paths=paths,
        save=True,
    )

    expected = (
        paths.processed / "predictions" / "2026" / "round_06_prediction.csv"
    )
    assert result.prediction_path == expected
    assert expected.exists()
    assert pd.read_csv(expected)["Driver"].tolist() == ["AAA"]
    assert captured["calibration_predictions"] is calibration.predictions
    assert not expected.with_suffix(".csv.tmp").exists()
    assert (paths.processed / "pre_race" / "2026" / "round_06_weekend_features.csv").exists()


def test_pre_race_prediction_preserves_existing_file(tmp_path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    destination = paths.processed / "predictions" / "2026" / "round_06_prediction.csv"
    destination.parent.mkdir(parents=True)
    destination.write_text("original", encoding="utf-8")

    with pytest.raises(prediction.PreRacePredictionError, match="zaten kayıtlı"):
        prediction.create_pre_race_prediction(2026, 6, paths=paths)

    assert destination.read_text(encoding="utf-8") == "original"


def test_pre_race_prediction_rejects_started_race(tmp_path, monkeypatch) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    monkeypatch.setattr(
        prediction,
        "load_history_before_round",
        lambda paths, season, prediction_round: pd.DataFrame({"RoundNumber": [1]}),
    )
    weekend = SimpleNamespace(
        race_start_utc=pd.Timestamp("2000-01-01", tz="UTC"),
    )
    monkeypatch.setattr(
        prediction,
        "collect_pre_race_weekend",
        lambda season, round_number, paths, save: weekend,
    )

    with pytest.raises(prediction.PreRacePredictionError, match="Yarış başlamış"):
        prediction.create_pre_race_prediction(2026, 2, paths=paths, save=False)


def test_started_race_does_not_write_weekend_features(tmp_path, monkeypatch) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    monkeypatch.setattr(
        prediction,
        "load_history_before_round",
        lambda paths, season, prediction_round: pd.DataFrame({"RoundNumber": [1]}),
    )
    weekend = SimpleNamespace(
        features=pd.DataFrame({"DriverId": ["A"]}),
        race_start_utc=pd.Timestamp("2000-01-01", tz="UTC"),
    )
    saves: list[bool] = []
    monkeypatch.setattr(
        prediction,
        "collect_pre_race_weekend",
        lambda season, round_number, paths, save: saves.append(save) or weekend,
    )

    with pytest.raises(prediction.PreRacePredictionError, match="Yarış başlamış"):
        prediction.create_pre_race_prediction(2026, 2, paths=paths, save=True)

    assert saves == [False]
    assert not (paths.processed / "pre_race" / "2026" / "round_02_weekend_features.csv").exists()


def test_retrospective_uses_prior_history_then_official_result(tmp_path, monkeypatch) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    calls: list[str] = []
    history = pd.DataFrame({"Season": [2026], "RoundNumber": [1]})
    weekend = SimpleNamespace(
        event_name="Test GP",
        features=pd.DataFrame({"Season": [2026], "RoundNumber": [2], "DriverId": ["A"]}),
        race_start_utc=pd.Timestamp("2000-01-02", tz="UTC"),
        final_grid=SimpleNamespace(
            document=SimpleNamespace(published_at=pd.Timestamp("2000-01-01", tz="UTC"))
        ),
    )
    output = pd.DataFrame({
        "DriverId": ["A"], "Driver": ["AAA"], "Team": ["Ferrari"],
        "StartOrder": [2], "PredictedPosition": [2], "EventName": ["Test GP"],
    })
    actual = pd.DataFrame({
        "Season": [2026], "RoundNumber": [2], "DriverId": ["A"],
        "ActualPosition": [1], "CleanMAEEligible": [True],
    })
    monkeypatch.setattr(
        prediction, "backfill_training_history",
        lambda season, through_round, paths: calls.append(f"backfill:{through_round}"),
    )
    monkeypatch.setattr(
        prediction, "load_history_before_round",
        lambda paths, season, prediction_round: calls.append("history") or history,
    )
    monkeypatch.setattr(
        prediction, "collect_pre_race_weekend",
        lambda season, round_number, paths, save: calls.append("weekend") or weekend,
    )

    def fake_predict(prior, features, **kwargs):
        assert prior["RoundNumber"].max() < features["RoundNumber"].min()
        assert "ActualPosition" not in features
        calls.append("predict")
        return UpcomingPredictionResult(output, 0.5, 1, 1)

    monkeypatch.setattr(prediction, "predict_upcoming_round", fake_predict)
    monkeypatch.setattr(
        prediction, "collect_completed_round",
        lambda season, round_number, paths, save, pre_race: (
            calls.append("result") or SimpleNamespace(rows=actual)
        ),
    )

    result = prediction.create_retrospective_prediction(2026, 2, paths=paths)

    assert calls == ["backfill:1", "history", "weekend", "predict", "result"]
    assert result.mae == 1.0
    assert result.clean_mae == 1.0
    assert result.prediction_path.exists()
    assert "retrospectives" in result.prediction_path.parts
    assert not (paths.processed / "predictions" / "2026" / "round_02_prediction.csv").exists()


def test_retrospective_rejects_grid_published_after_start(tmp_path, monkeypatch) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    monkeypatch.setattr(prediction, "backfill_training_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prediction, "load_history_before_round",
        lambda *args, **kwargs: pd.DataFrame({"RoundNumber": [1]}),
    )
    monkeypatch.setattr(
        prediction, "collect_pre_race_weekend",
        lambda *args, **kwargs: SimpleNamespace(
            race_start_utc=pd.Timestamp("2000-01-01", tz="UTC"),
            final_grid=SimpleNamespace(
                document=SimpleNamespace(published_at=pd.Timestamp("2000-01-02", tz="UTC"))
            ),
        ),
    )
    with pytest.raises(prediction.PreRacePredictionError, match="gridinin yarıştan önce"):
        prediction.create_retrospective_prediction(2026, 2, paths=paths)


def test_retrospective_preserves_real_pre_race_prediction(tmp_path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    live = paths.processed / "predictions" / "2026" / "round_02_prediction.csv"
    live.parent.mkdir(parents=True)
    live.write_text("original", encoding="utf-8")

    with pytest.raises(prediction.PreRacePredictionError, match="gerçek yarış öncesi"):
        prediction.create_retrospective_prediction(2026, 2, paths=paths)

    assert live.read_text(encoding="utf-8") == "original"
