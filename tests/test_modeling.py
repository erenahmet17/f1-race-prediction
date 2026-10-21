from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import f1_prediction.modeling as modeling

from f1_prediction.modeling import (
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_GRID_DELTA_WEIGHT,
    add_prior_form_features,
    build_model_features,
    grid_delta_scores,
    penalty_aware_hybrid_scores,
    predict_upcoming_round,
    rank_scores,
    select_grid_delta_weight,
)


def history_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_number, positions in ((1, (2, 1)), (2, (1, 2))):
        for start_order, (driver, team, actual) in enumerate(
            zip(("AAA", "BBB"), ("One", "Two"), positions, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "Season": 2026,
                    "RoundNumber": round_number,
                    "DriverId": f"abbr:{driver}",
                    "Driver": driver,
                    "Team": team,
                    "StartOrder": start_order,
                    "StartOrderPct": start_order - 1,
                    "ActualPosition": actual,
                    "ActualPositionPct": actual - 1,
                    "CleanMAEEligible": True,
                    "RaceDNF": False,
                }
            )
    return pd.DataFrame(rows)


def test_prior_form_uses_only_earlier_rounds() -> None:
    enriched = add_prior_form_features(history_rows())
    round_one = enriched[enriched["RoundNumber"] == 1]
    round_two = enriched[enriched["RoundNumber"] == 2].set_index("Driver")

    assert round_one["DriverPriorRoundCount"].eq(0).all()
    assert round_two.loc["AAA", "DriverPriorRoundCount"] == 1
    assert round_two.loc["AAA", "DriverPriorMeanFinishPct"] == 1
    assert round_two.loc["BBB", "DriverPriorMeanFinishPct"] == 0


def test_rank_scores_breaks_ties_with_final_grid() -> None:
    ranks = rank_scores(
        np.array([0.5, 0.5, 0.7]),
        start_order=pd.Series([2, 1, 3]),
        driver_ids=pd.Series(["A", "B", "C"]),
    )

    assert ranks.tolist() == [3, 2, 1]


def test_grid_delta_scores_can_promote_last_grid_driver_without_cap() -> None:
    start_order = pd.Series([1, 2, 3])
    start_order_pct = pd.Series([0.0, 0.5, 1.0])
    scores = grid_delta_scores(
        np.array([-0.20, 0.0, 0.95]),
        start_order_pct=start_order_pct,
    )

    ranks = rank_scores(
        scores,
        start_order=start_order,
        driver_ids=pd.Series(["A", "B", "C"]),
    )

    assert ranks.tolist() == [2, 3, 1]


def test_grid_delta_weight_defaults_without_past_predictions() -> None:
    assert select_grid_delta_weight(pd.DataFrame()) == DEFAULT_GRID_DELTA_WEIGHT


def test_grid_delta_weight_uses_only_supplied_past_predictions() -> None:
    previous = pd.DataFrame(
        {
            "RoundNumber": [5, 5, 5],
            "DriverId": ["A", "B", "C"],
            "StartOrder": [1, 2, 3],
            "ActualPosition": [1, 2, 3],
            "ExpectedPositionsGainedPct": [0.0, 3.0, -2.0],
        }
    )

    assert select_grid_delta_weight(previous) == 0.0


def test_penalty_aware_hybrid_uses_direct_rank_only_for_penalized() -> None:
    scores = penalty_aware_hybrid_scores(
        catboost_ranks=pd.Series([1, 3, 2]),
        grid_delta_ranks=pd.Series([3, 1, 2]),
        penalty_applied=pd.Series([True, False, False]),
    )

    assert scores.tolist() == [-1.0, -1.0, -2.0]


def test_model_matrix_has_explicit_pre_race_allowlist() -> None:
    base = history_rows().iloc[[0]].copy()
    for column in CATEGORICAL_FEATURES:
        if column not in base:
            base[column] = "value"
    for column in NUMERIC_FEATURES:
        if column not in base:
            base[column] = np.nan
    base["QualifyingPositionToTeammate"] = 0
    base["TeamQualifyingPositionMean"] = 1
    base["SprintStartPosition"] = np.nan
    base["SprintFinishPosition"] = np.nan
    base["SprintPositionsGained"] = np.nan
    base["GridPenaltyPositions"] = np.nan
    base["RaceStatus"] = "Finished"

    features = build_model_features(base)

    assert tuple(features.columns) == MODEL_FEATURES
    assert not FORBIDDEN_MODEL_COLUMNS.intersection(features.columns)


def model_ready_rows(*, rounds: range) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_number in rounds:
        for start_order, (driver, team) in enumerate(
            (("AAA", "One"), ("BBB", "Two")),
            start=1,
        ):
            actual = start_order if round_number % 2 else 3 - start_order
            row: dict[str, object] = {
                "Season": 2026,
                "RoundNumber": round_number,
                "EventName": f"Round {round_number}",
                "DriverId": f"abbr:{driver}",
                "Driver": driver,
                "Team": team,
                "QualifyingPosition": start_order,
                "StartOrder": start_order,
                "StartOrderPct": start_order - 1,
                "GridVsQualifyingDelta": 0,
                "GridPenaltyApplied": False,
                "GridPenaltyPositions": np.nan,
                "GridPenaltyReason": "NONE",
                "ActualPosition": actual,
                "ActualPositionPct": actual - 1,
                "CleanMAEEligible": True,
                "RaceDNF": False,
                "QualifyingPositionToTeammate": 0,
                "TeamQualifyingPositionMean": start_order,
                "SprintStartPosition": np.nan,
                "SprintFinishPosition": np.nan,
                "SprintPositionsGained": np.nan,
            }
            for column in CATEGORICAL_FEATURES:
                row.setdefault(column, "VALUE")
            for column in NUMERIC_FEATURES:
                row.setdefault(column, 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_upcoming_prediction_trains_only_on_completed_rounds(monkeypatch) -> None:
    history = model_ready_rows(rounds=range(1, 6))
    upcoming = model_ready_rows(rounds=range(6, 7)).drop(
        columns=[
            "ActualPosition",
            "ActualPositionPct",
            "CleanMAEEligible",
            "RaceDNF",
        ]
    )
    monkeypatch.setattr(
        modeling,
        "_ridge_scores",
        lambda train_x, train_y, test_x: np.array([0.8, 0.2]),
    )
    monkeypatch.setattr(
        modeling,
        "_catboost_scores",
        lambda train_x, train_y, train_groups, test_x: np.array([0.8, 0.2]),
    )
    monkeypatch.setattr(
        modeling,
        "_catboost_grid_delta_scores",
        lambda train_x, train_y, test_x: np.array([0.0, 0.0]),
    )

    result = predict_upcoming_round(history, upcoming, min_train_rounds=4)

    assert result.training_rounds == 5
    assert result.training_rows == 10
    assert result.predictions["TrainingThroughRound"].eq(5).all()
    assert result.predictions["PredictedPosition"].tolist() == [1, 2]
    assert "ActualPosition" not in result.predictions


def test_upcoming_prediction_rejects_race_result_leakage() -> None:
    history = model_ready_rows(rounds=range(1, 6))
    upcoming = model_ready_rows(rounds=range(6, 7))

    with pytest.raises(modeling.ModelingError, match="yarış sonrası sütunu"):
        predict_upcoming_round(history, upcoming, min_train_rounds=4)
