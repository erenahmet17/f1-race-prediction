from __future__ import annotations

import numpy as np
import pandas as pd

from f1_prediction.practice import build_practice_features, clean_practice_laps


def lap(
    driver: str,
    team: str,
    number: int,
    seconds: float,
    *,
    stint: int = 1,
    compound: str = "MEDIUM",
    tyre_life: int = 5,
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
        "Compound": compound,
        "TyreLife": tyre_life,
        "Stint": stint,
        "PitOutTime": pd.NaT,
        "PitInTime": pd.Timedelta(seconds=seconds) if pit_in else pd.NaT,
        "IsAccurate": accurate,
        "Deleted": deleted,
        "TrackStatus": track_status,
    }


def test_clean_practice_laps_removes_invalid_laps() -> None:
    laps = pd.DataFrame(
        [
            lap("AAA", "One", 1, 90),
            lap("AAA", "One", 2, 90, pit_in=True),
            lap("AAA", "One", 3, 90, deleted=True),
            lap("AAA", "One", 4, 90, accurate=False),
            lap("AAA", "One", 5, 90, track_status="4"),
        ]
    )

    clean = clean_practice_laps(laps)

    assert clean["LapNumber"].tolist() == [1]


def test_missing_deleted_flag_means_lap_was_not_deleted() -> None:
    laps = pd.DataFrame([lap("AAA", "One", 1, 90)])
    laps["Deleted"] = pd.Series([None], dtype="object")

    clean = clean_practice_laps(laps)

    assert clean["LapNumber"].tolist() == [1]


def test_short_stint_is_marked_missing() -> None:
    laps = pd.DataFrame(
        [lap("AAA", "One", number, 90) for number in range(1, 5)]
        + [lap("BBB", "Two", number, 95) for number in range(1, 7)]
    )

    features = build_practice_features({"FP1": laps}, expected_sessions=1)
    drivers = features.drivers.set_index("Driver")

    assert drivers.loc["AAA", "PracticeDataQuality"] == "MISSING"
    assert np.isnan(drivers.loc["AAA", "PracticePaceDeltaPct"])
    assert drivers.loc["BBB", "PracticeLongRunLapCount"] == 6


def test_relative_pace_uses_comparable_laps() -> None:
    rows = []
    for number in range(1, 7):
        rows.extend(
            [
                lap("AAA", "One", number, 90, tyre_life=number),
                lap("BBB", "One", number, 92, tyre_life=number),
                lap("CCC", "Two", number, 94, tyre_life=number),
            ]
        )

    features = build_practice_features(
        {"FP1": pd.DataFrame(rows)}, expected_sessions=1
    )
    drivers = features.drivers.set_index("Driver")

    assert drivers.loc["AAA", "PracticePaceDeltaPct"] < 0
    assert np.isclose(drivers.loc["BBB", "PracticePaceDeltaPct"], 0.0)
    assert drivers.loc["CCC", "PracticePaceDeltaPct"] > 0
    assert drivers.loc["AAA", "PracticeReferenceFallbackPct"] == 0.0


def test_cooldown_laps_do_not_distort_long_run_reference() -> None:
    rows = []
    for driver, team, pace in (
        ("AAA", "One", 90),
        ("BBB", "Two", 92),
        ("CCC", "Three", 94),
    ):
        for number in range(1, 7):
            rows.append(lap(driver, team, number, pace, tyre_life=number))
        rows.append(lap(driver, team, 7, 125, tyre_life=7))

    features = build_practice_features(
        {"FP1": pd.DataFrame(rows)}, expected_sessions=1
    )
    drivers = features.drivers.set_index("Driver")

    assert drivers.loc["AAA", "PracticeLongRunLapCount"] == 6
    assert np.isclose(drivers.loc["BBB", "PracticePaceDeltaPct"], 0.0)


def test_wet_and_dry_laps_use_separate_speed_thresholds() -> None:
    rows = []
    for driver, team, dry_pace, wet_pace in (
        ("AAA", "One", 90, 120),
        ("BBB", "Two", 92, 122),
        ("CCC", "Three", 94, 124),
    ):
        for number in range(1, 7):
            rows.append(lap(driver, team, number, dry_pace, stint=1))
        for number in range(7, 13):
            rows.append(
                lap(
                    driver,
                    team,
                    number,
                    wet_pace,
                    stint=2,
                    compound="INTERMEDIATE",
                )
            )

    features = build_practice_features(
        {"FP1": pd.DataFrame(rows)}, expected_sessions=1
    )
    drivers = features.drivers.set_index("Driver")

    assert drivers.loc["AAA", "PracticeLongRunLapCount"] == 12
    assert drivers.loc["BBB", "PracticeLongRunLapCount"] == 12


def test_reserve_driver_contributes_to_team_pace() -> None:
    fp1 = pd.DataFrame(
        [lap("RES", "One", number, 90) for number in range(1, 7)]
        + [lap("BBB", "Two", number, 92) for number in range(1, 7)]
        + [lap("CCC", "Three", number, 94) for number in range(1, 7)]
    )

    features = build_practice_features({"FP1": fp1}, expected_sessions=1)
    teams = features.teams.set_index("Team")

    assert "RES" in features.drivers["Driver"].tolist()
    assert teams.loc["One", "TeamPracticeDriverCount"] == 1
    assert teams.loc["One", "TeamPracticePaceDeltaPct"] < 0
