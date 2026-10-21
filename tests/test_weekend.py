from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_prediction.weekend import WeekendFeatureError, build_weekend_feature_table


def qualifying() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "DriverId": "aaa",
                "Driver": "AAA",
                "Team": "One",
                "QualifyingPosition": 1,
                "QualiTimeMissing": False,
            },
            {
                "DriverId": "bbb",
                "Driver": "BBB",
                "Team": "Two",
                "QualifyingPosition": 2,
                "QualiTimeMissing": False,
            },
        ]
    )
    frame["QualifyingPositionPct"] = [0.0, 1.0]
    frame["QualiTime_s"] = [90.0, 91.0]
    frame["QualiGapPct"] = [0.0, 1.11]
    frame["Q1GapPct"] = [0.0, 1.0]
    frame["Q2GapPct"] = [0.0, 1.0]
    frame["Q3GapPct"] = [0.0, 1.0]
    frame["QualiGapToTeammatePct"] = np.nan
    frame["QualifyingPositionToTeammate"] = 0.0
    frame["TeamQualifyingPositionMean"] = [1.0, 2.0]
    frame["QualiStageReached"] = "Q3"
    frame["QualifyingDataQuality"] = "USABLE"
    return frame


def final_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverId": "bbb",
                "CarNumber": "2",
                "GridPosition": 1,
                "StartsFromPitLane": False,
                "StartOrder": 1,
                "StartOrderPct": 0.0,
                "GridStatus": "FINAL",
                "GridSource": "https://fia.example/final.pdf",
                "GridPublishedAt": pd.Timestamp("2026-09-20T14:00:00+02:00"),
                "GridDocumentNumber": "42",
            },
            {
                "DriverId": "aaa",
                "CarNumber": "1",
                "GridPosition": 2,
                "StartsFromPitLane": False,
                "StartOrder": 2,
                "StartOrderPct": 1.0,
                "GridStatus": "FINAL",
                "GridSource": "https://fia.example/final.pdf",
                "GridPublishedAt": pd.Timestamp("2026-09-20T14:00:00+02:00"),
                "GridDocumentNumber": "42",
            },
        ]
    )


def practice_drivers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": team,
                "PracticePaceDeltaPct": pace,
                "PracticeConsistencyPct": 0.2,
                "PracticeLongRunLapCount": 10,
                "PracticeStintCount": 2,
                "PracticeSessionsUsed": 2,
                "PracticeReferenceFallbackPct": 0.0,
                "PracticeDataReliability": 1.0,
                "PracticeDataQuality": "USABLE",
                "PracticeDataMissing": False,
            }
            for driver, team, pace in (
                ("AAA", "One", -0.5),
                ("BBB", "Two", 0.5),
                ("RES", "One", -0.7),
            )
        ]
    )


def practice_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Team": "One",
                "TeamPracticePaceDeltaPct": -0.6,
                "TeamPracticeDriverCount": 2,
                "TeamPracticeLongRunLapCount": 20,
                "TeamPracticeDataReliability": 1.0,
            },
            {
                "Team": "Two",
                "TeamPracticePaceDeltaPct": 0.5,
                "TeamPracticeDriverCount": 1,
                "TeamPracticeLongRunLapCount": 10,
                "TeamPracticeDataReliability": 1.0,
            },
        ]
    )


def sprint() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": team,
                "SprintPaceDeltaPct": pace,
                "SprintConsistencyPct": 0.1,
                "SprintCleanLapCount": 12,
                "SprintDataReliability": 1.0,
                "SprintDataQuality": "USABLE",
                "SprintStartPosition": position,
                "SprintFinishPosition": position,
                "SprintStatus": "Finished",
                "SprintLapsCompleted": 20,
                "SprintPositionsGained": 0,
                "SprintLapsCompletedPct": 1.0,
                "SprintDNF": False,
                "TeamSprintPaceDeltaPct": pace,
            }
            for driver, team, pace, position in (
                ("AAA", "One", -0.4, 1),
                ("BBB", "Two", 0.4, 2),
            )
        ]
    )


def test_normal_weekend_merges_on_qualifying_drivers_and_final_start_order() -> None:
    features = build_weekend_feature_table(
        qualifying(),
        final_grid(),
        season=2026,
        round_number=14,
        event_name="Example Grand Prix",
        practice_drivers=practice_drivers(),
        practice_teams=practice_teams(),
    )

    assert features["DriverId"].tolist() == ["bbb", "aaa"]
    assert "RES" not in features["Driver"].tolist()
    assert features.set_index("Driver").loc["AAA", "TeamPracticeDriverCount"] == 2
    assert features["SprintDataQuality"].eq("NOT_APPLICABLE").all()
    assert not features["SprintDataMissing"].any()
    assert "SprintFinishPosition" in features.columns
    assert features["SprintFinishPosition"].isna().all()
    assert features["Season"].eq(2026).all()
    by_driver = features.set_index("Driver")
    assert by_driver.loc["AAA", "GridVsQualifyingDelta"] == 1
    assert by_driver.loc["AAA", "GridVsQualifyingDeltaPct"] == 1.0
    assert by_driver["GridPenaltyReason"].eq("NONE").all()


def test_missing_practice_stays_explicit_without_pace_imputation() -> None:
    drivers = practice_drivers().query("Driver == 'AAA'").copy()
    features = build_weekend_feature_table(
        qualifying(),
        final_grid(),
        season=2026,
        round_number=14,
        event_name="Example Grand Prix",
        practice_drivers=drivers,
        practice_teams=practice_teams(),
    ).set_index("Driver")

    assert np.isnan(features.loc["BBB", "PracticePaceDeltaPct"])
    assert features.loc["BBB", "PracticeDataQuality"] == "MISSING"
    assert features.loc["BBB", "PracticeDataReliability"] == 0.0
    assert bool(features.loc["BBB", "PracticeDataMissing"])
    assert not bool(features.loc["BBB", "PracticeDriverDataAvailable"])


def test_sprint_weekend_merges_sprint_results() -> None:
    features = build_weekend_feature_table(
        qualifying(),
        final_grid(),
        season=2026,
        round_number=14,
        event_name="Example Grand Prix",
        sprint=sprint(),
        is_sprint_weekend=True,
    ).set_index("Driver")

    assert features["SprintWeekend"].all()
    assert features["SprintDataAvailable"].all()
    assert features.loc["AAA", "SprintFinishPosition"] == 1
    assert not features["SprintDataMissing"].any()


def test_sprint_weekend_without_sprint_data_is_rejected() -> None:
    with pytest.raises(WeekendFeatureError, match="sprint özellikleri gerekli"):
        build_weekend_feature_table(
            qualifying(),
            final_grid(),
            season=2026,
            round_number=14,
            event_name="Example Grand Prix",
            is_sprint_weekend=True,
        )


def test_final_grid_must_match_qualifying_drivers() -> None:
    invalid_grid = final_grid().query("DriverId == 'aaa'").copy()

    with pytest.raises(WeekendFeatureError, match="qualifying ile uyuşmuyor"):
        build_weekend_feature_table(
            qualifying(),
            invalid_grid,
            season=2026,
            round_number=14,
            event_name="Example Grand Prix",
        )


def test_team_mismatch_rejects_mixed_event_data() -> None:
    invalid_practice = practice_drivers().copy()
    invalid_practice.loc[invalid_practice["Driver"] == "AAA", "Team"] = "Wrong"

    with pytest.raises(WeekendFeatureError, match="qualifying ile uyuşmuyor"):
        build_weekend_feature_table(
            qualifying(),
            final_grid(),
            season=2026,
            round_number=14,
            event_name="Example Grand Prix",
            practice_drivers=invalid_practice,
        )
