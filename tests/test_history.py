from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.collection import WeekendDataPaths
from f1_prediction.history import (
    HistoryDataError,
    InsufficientHistoryError,
    build_completed_round_rows,
    load_history_before_round,
    select_history_before_round,
)
from f1_prediction.race import RACE_TARGET_COLUMNS


def pre_race() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Season": 2026,
                "RoundNumber": 2,
                "DriverId": "aaa",
                "Driver": "AAA",
                "Team": "One",
                "StartOrder": 1,
                "QualiGapPct": 0.0,
            },
            {
                "Season": 2026,
                "RoundNumber": 2,
                "DriverId": "bbb",
                "Driver": "BBB",
                "Team": "Two",
                "StartOrder": 2,
                "QualiGapPct": 1.0,
            },
        ]
    )


def targets() -> pd.DataFrame:
    rows = []
    for driver_id, driver, team, position in (
        ("aaa", "AAA", "One", 2),
        ("bbb", "BBB", "Two", 1),
    ):
        row: dict[str, object] = {
            "DriverId": driver_id,
            "RaceDriver": driver,
            "RaceTeam": team,
        }
        row.update({column: 0 for column in RACE_TARGET_COLUMNS})
        row["ActualPosition"] = position
        row["ActualPositionPct"] = position - 1
        row["CleanMAEEligible"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def test_completed_round_keeps_targets_separate_from_pre_race_columns() -> None:
    rows = build_completed_round_rows(pre_race(), targets())

    assert rows["DriverId"].tolist() == ["aaa", "bbb"]
    assert rows.set_index("DriverId").loc["bbb", "ActualPosition"] == 1
    assert "RaceDriver" not in rows.columns


def test_mixed_driver_identity_is_rejected() -> None:
    invalid = targets()
    invalid.loc[invalid["DriverId"] == "aaa", "RaceTeam"] = "Wrong"

    with pytest.raises(HistoryDataError, match="kimlikleri"):
        build_completed_round_rows(pre_race(), invalid)


def test_select_history_excludes_current_and_future_rounds() -> None:
    history = pd.DataFrame(
        [
            {"Season": 2026, "RoundNumber": round_number, "DriverId": "aaa", "ActualPosition": 1}
            for round_number in range(1, 7)
        ]
        + [
            {"Season": 2025, "RoundNumber": 1, "DriverId": "old", "ActualPosition": 1}
        ]
    )

    selected = select_history_before_round(
        history, season=2026, prediction_round=5
    )

    assert selected["RoundNumber"].unique().tolist() == [1, 2, 3, 4]
    assert selected["Season"].unique().tolist() == [2026]


def test_load_history_requires_every_previous_round(tmp_path: Path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    directory = paths.processed / "history" / "2026"
    directory.mkdir(parents=True)
    frame = pd.DataFrame(
        [{"Season": 2026, "RoundNumber": 1, "DriverId": "aaa", "ActualPosition": 1}]
    )
    frame.to_csv(directory / "round_01_training_rows.csv", index=False)

    with pytest.raises(InsufficientHistoryError, match="2, 3"):
        load_history_before_round(paths, season=2026, prediction_round=4)


def test_load_history_ignores_files_from_prediction_round_and_later(
    tmp_path: Path,
) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    directory = paths.processed / "history" / "2026"
    directory.mkdir(parents=True)
    for round_number in range(1, 7):
        frame = pd.DataFrame(
            [
                {
                    "Season": 2026,
                    "RoundNumber": round_number,
                    "DriverId": "aaa",
                    "ActualPosition": 1,
                }
            ]
        )
        frame.to_csv(
            directory / f"round_{round_number:02d}_training_rows.csv", index=False
        )

    history = load_history_before_round(paths, season=2026, prediction_round=5)

    assert history["RoundNumber"].unique().tolist() == [1, 2, 3, 4]


def test_first_round_prediction_is_explicitly_blocked() -> None:
    with pytest.raises(InsufficientHistoryError, match="ilk yarışı"):
        select_history_before_round(
            pd.DataFrame(), season=2026, prediction_round=1
        )
