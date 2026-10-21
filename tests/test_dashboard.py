from __future__ import annotations

import pandas as pd

from f1_prediction.dashboard import (
    available_rounds,
    available_seasons,
    build_race_view,
    style_team_rows,
)


def test_saved_prediction_takes_priority_over_backtest_and_uses_real_result() -> None:
    saved = pd.DataFrame(
        {
            "DriverId": ["A", "B"],
            "Driver": ["AAA", "BBB"],
            "Team": ["One", "Two"],
            "StartOrder": [1, 2],
            "PredictedPosition": [2, 1],
            "EventName": ["Test GP", "Test GP"],
        }
    )
    backtest = saved.drop(columns="PredictedPosition").assign(
        PenaltyAwareHybridPredictedRank=[1, 2]
    )
    history = pd.DataFrame(
        {
            "DriverId": ["A", "B"],
            "ActualPosition": [1, 2],
            "CleanMAEEligible": [True, False],
        }
    )

    view = build_race_view(
        saved_prediction=saved,
        backtest_prediction=backtest,
        history=history,
    )

    assert view.source == "Kaydedilmiş yarış öncesi tahmin"
    assert view.rows["Driver"].tolist() == ["BBB", "AAA"]
    assert view.mae == 1.0
    assert view.clean_mae == 1.0


def test_no_prediction_shows_result_without_invented_mae() -> None:
    history = pd.DataFrame(
        {
            "DriverId": ["A"],
            "Driver": ["AAA"],
            "Team": ["One"],
            "StartOrder": [2],
            "ActualPosition": [1],
            "EventName": ["Test GP"],
        }
    )

    view = build_race_view(
        saved_prediction=pd.DataFrame(),
        backtest_prediction=pd.DataFrame(),
        history=history,
    )

    assert view.source is None
    assert view.mae is None
    assert "PredictedPosition" not in view.rows


def test_seasons_and_rounds_include_live_prediction_files(tmp_path) -> None:
    live = tmp_path / "predictions" / "2026"
    live.mkdir(parents=True)
    (live / "round_15_prediction.csv").write_text("Driver\nAAA\n", encoding="utf-8")

    assert available_seasons(tmp_path) == [2026]
    assert available_rounds(tmp_path, 2026, pd.DataFrame()) == [15]


def test_team_rows_use_team_colors_and_unknown_team_stays_neutral() -> None:
    frame = pd.DataFrame({"Takım": ["McLaren", "Ferrari", "Unknown"], "Pilot": ["A", "B", "C"]})
    styled = style_team_rows(frame, team_column="Takım")._compute()

    assert (0, 0) in styled.ctx and "#754013" in str(styled.ctx[(0, 0)])
    assert (1, 1) in styled.ctx and "#75242c" in str(styled.ctx[(1, 1)])
    assert (2, 0) not in styled.ctx


def test_retrospective_is_separate_and_precedes_backtest(tmp_path) -> None:
    retro = tmp_path / "retrospectives" / "2026"
    retro.mkdir(parents=True)
    (retro / "round_15_retrospective.csv").write_text("Driver\nAAA\n", encoding="utf-8")
    assert available_seasons(tmp_path) == [2026]
    assert available_rounds(tmp_path, 2026, pd.DataFrame()) == [15]

    prediction = pd.DataFrame({
        "DriverId": ["A"], "Driver": ["AAA"], "Team": ["Ferrari"],
        "StartOrder": [2], "PredictedPosition": [2], "EventName": ["Test GP"],
    })
    backtest = prediction.drop(columns="PredictedPosition").assign(
        PenaltyAwareHybridPredictedRank=[1]
    )
    history = pd.DataFrame({
        "DriverId": ["A"], "ActualPosition": [1], "CleanMAEEligible": [True]
    })
    view = build_race_view(
        saved_prediction=pd.DataFrame(),
        retrospective_prediction=prediction,
        backtest_prediction=backtest,
        history=history,
    )
    assert view.source == "Yarış sonrası geriye dönük deneme"
    assert view.mae == 1.0
