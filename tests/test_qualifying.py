from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_prediction.qualifying import QualifyingDataError, build_qualifying_features


def results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverId": "aaa",
                "Abbreviation": "AAA",
                "TeamName": "One",
                "Position": 1,
                "Q1": pd.Timedelta(seconds=91),
                "Q2": pd.Timedelta(seconds=90.5),
                "Q3": pd.Timedelta(seconds=90),
            },
            {
                "DriverId": "bbb",
                "Abbreviation": "BBB",
                "TeamName": "One",
                "Position": 2,
                "Q1": pd.Timedelta(seconds=92),
                "Q2": pd.Timedelta(seconds=91),
                "Q3": pd.Timedelta(seconds=90.9),
            },
            {
                "DriverId": "ccc",
                "Abbreviation": "CCC",
                "TeamName": "Two",
                "Position": 3,
                "Q1": pd.NaT,
                "Q2": pd.NaT,
                "Q3": pd.NaT,
            },
        ]
    )


def test_qualifying_times_are_relative_to_pole() -> None:
    features = build_qualifying_features(results()).set_index("DriverId")

    assert np.isclose(features.loc["aaa", "QualiGapPct"], 0.0)
    assert np.isclose(features.loc["bbb", "QualiGapPct"], 1.0)
    assert np.isclose(features.loc["bbb", "QualiGapToTeammatePct"], 1.0)
    assert features.loc["bbb", "QualiStageReached"] == "Q3"


def test_driver_without_time_remains_missing() -> None:
    features = build_qualifying_features(results()).set_index("DriverId")

    assert np.isnan(features.loc["ccc", "QualiTime_s"])
    assert bool(features.loc["ccc", "QualiTimeMissing"])
    assert features.loc["ccc", "QualifyingDataQuality"] == "MISSING"


def test_no_valid_time_stops_processing() -> None:
    no_times = results()
    no_times[["Q1", "Q2", "Q3"]] = pd.NaT

    with pytest.raises(QualifyingDataError, match="geçerli qualifying süresi yok"):
        build_qualifying_features(no_times)


def test_last_reached_stage_is_used_instead_of_overall_best_lap() -> None:
    data = results().iloc[:2].copy()
    data.loc[data["DriverId"] == "bbb", "Q1"] = pd.Timedelta(seconds=89)
    data.loc[data["DriverId"] == "bbb", "Q3"] = pd.Timedelta(seconds=91)

    features = build_qualifying_features(data).set_index("DriverId")

    assert np.isclose(features.loc["bbb", "QualiTime_s"], 91.0)
    assert features.loc["bbb", "QualiGapPct"] > 0
