from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.race import RaceResultError, build_race_targets


def grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"DriverId": "aaa", "StartOrder": 1},
            {"DriverId": "bbb", "StartOrder": 2},
            {"DriverId": "ccc", "StartOrder": 3},
            {"DriverId": "ddd", "StartOrder": 4},
        ]
    )


def results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverId": "bbb",
                "Abbreviation": "BBB",
                "TeamName": "Two",
                "Position": 1,
                "Status": "Finished",
                "Laps": 50,
            },
            {
                "DriverId": "aaa",
                "Abbreviation": "AAA",
                "TeamName": "One",
                "Position": 2,
                "Status": "Lapped",
                "Laps": 49,
            },
            {
                "DriverId": "ccc",
                "Abbreviation": "CCC",
                "TeamName": "Three",
                "Position": 3,
                "Status": "Retired",
                "Laps": 20,
            },
            {
                "DriverId": "ddd",
                "Abbreviation": "DDD",
                "TeamName": "Four",
                "Position": 4,
                "Status": "Disqualified",
                "Laps": 50,
            },
        ]
    )


def official_classification() -> pd.DataFrame:
    return results()[["DriverId", "Position"]].rename(
        columns={"Position": "RaceOfficialPosition"}
    )


def test_official_positions_and_clean_mae_flags_are_preserved() -> None:
    targets = build_race_targets(
        results(), grid(), official_classification()
    ).set_index("DriverId")

    assert targets.loc["bbb", "ActualPosition"] == 1
    assert bool(targets.loc["aaa", "CleanMAEEligible"])
    assert bool(targets.loc["ccc", "RaceDNF"])
    assert not bool(targets.loc["ccc", "CleanMAEEligible"])
    assert bool(targets.loc["ddd", "RaceDSQ"])


def test_missing_positions_use_unused_places_in_final_grid_order() -> None:
    official = official_classification()
    official.loc[
        official["DriverId"].isin(["aaa", "ccc"]), "RaceOfficialPosition"
    ] = None

    targets = build_race_targets(results(), grid(), official).set_index("DriverId")

    assert targets.loc["aaa", "ActualPosition"] == 2
    assert targets.loc["ccc", "ActualPosition"] == 3
    assert targets.loc["aaa", "ActualPositionSource"] == "GRID_FALLBACK"
    assert targets.loc["bbb", "ActualPositionSource"] == "FIA"


def test_driver_list_must_match_final_grid() -> None:
    with pytest.raises(RaceResultError, match="final grid ile uyuşmuyor"):
        build_race_targets(
            results().iloc[:3].copy(), grid(), official_classification()
        )


def test_duplicate_official_position_is_rejected() -> None:
    official = official_classification()
    official.loc[official["DriverId"] == "aaa", "RaceOfficialPosition"] = 1

    with pytest.raises(RaceResultError, match="tekrarlanan pozisyon"):
        build_race_targets(results(), grid(), official)
