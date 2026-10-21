from __future__ import annotations

import numpy as np
import pandas as pd

from f1_prediction.sprint import build_sprint_features, clean_sprint_laps


def lap(
    driver: str,
    team: str,
    number: int,
    seconds: float,
    *,
    accurate: bool = True,
    deleted: bool = False,
    track_status: str = "1",
    pit_in: bool = False,
) -> dict[str, object]:
    return {
        "Driver": driver,
        "Team": team,
        "LapNumber": number,
        "LapTime": pd.Timedelta(seconds=seconds),
        "PitOutTime": pd.NaT,
        "PitInTime": pd.Timedelta(seconds=seconds) if pit_in else pd.NaT,
        "IsAccurate": accurate,
        "Deleted": deleted,
        "TrackStatus": track_status,
    }


def test_clean_sprint_laps_removes_opening_and_invalid_laps() -> None:
    laps = pd.DataFrame(
        [
            lap("AAA", "One", 1, 100),
            lap("AAA", "One", 2, 99),
            lap("AAA", "One", 3, 98),
            lap("AAA", "One", 4, 97, pit_in=True),
            lap("AAA", "One", 5, 96, track_status="4"),
            lap("AAA", "One", 6, 95, deleted=True),
            lap("AAA", "One", 7, 94, accurate=False),
        ]
    )

    clean = clean_sprint_laps(laps)

    assert clean["LapNumber"].tolist() == [3]


def test_missing_deleted_flag_means_lap_was_not_deleted() -> None:
    laps = pd.DataFrame([lap("AAA", "One", 3, 90)])
    laps["Deleted"] = pd.Series([None], dtype="object")

    clean = clean_sprint_laps(laps)

    assert clean["LapNumber"].tolist() == [3]


def test_build_sprint_features_uses_same_lap_field_median() -> None:
    rows = []
    for number in range(3, 13):
        rows.extend(
            [
                lap("AAA", "One", number, 90),
                lap("BBB", "One", number, 100),
                lap("CCC", "Two", number, 110),
            ]
        )

    features = build_sprint_features(pd.DataFrame(rows))
    indexed = features.set_index("Driver")

    assert np.isclose(indexed.loc["BBB", "SprintPaceDeltaPct"], 0.0)
    assert indexed.loc["AAA", "SprintPaceDeltaPct"] < 0
    assert indexed.loc["CCC", "SprintPaceDeltaPct"] > 0
    assert indexed.loc["AAA", "SprintDataQuality"] == "USABLE"
    assert np.isclose(indexed.loc["AAA", "TeamSprintPaceDeltaPct"], -5.0)


def test_short_sprint_sample_is_marked_missing() -> None:
    rows = []
    for number in range(3, 6):
        rows.append(lap("AAA", "One", number, 90))
    for number in range(3, 13):
        rows.append(lap("BBB", "Two", number, 100))

    features = build_sprint_features(pd.DataFrame(rows)).set_index("Driver")

    assert features.loc["AAA", "SprintDataQuality"] == "MISSING"
    assert np.isnan(features.loc["AAA", "SprintPaceDeltaPct"])
    assert features.loc["BBB", "SprintDataQuality"] == "USABLE"


def test_results_add_start_finish_and_dnf_features() -> None:
    rows = []
    for number in range(3, 13):
        rows.extend(
            [
                lap("AAA", "One", number, 90),
                lap("BBB", "Two", number, 100),
            ]
        )
    results = pd.DataFrame(
        [
            {
                "Abbreviation": "AAA",
                "GridPosition": 2,
                "Position": 1,
                "Status": "Finished",
                "Laps": 20,
            },
            {
                "Abbreviation": "BBB",
                "GridPosition": 1,
                "Position": 2,
                "Status": "Retired",
                "Laps": 10,
            },
        ]
    )

    features = build_sprint_features(pd.DataFrame(rows), results).set_index("Driver")

    assert features.loc["AAA", "SprintPositionsGained"] == 1
    assert not bool(features.loc["AAA", "SprintDNF"])
    assert bool(features.loc["BBB", "SprintDNF"])
    assert np.isclose(features.loc["BBB", "SprintLapsCompletedPct"], 0.5)
