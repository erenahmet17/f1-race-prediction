from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from f1_prediction.collection import (
    WeekendCollectionError,
    WeekendDataPaths,
    _align_team_names,
    classify_event_sessions,
    collect_pre_race_weekend,
)
from f1_prediction.fia_grid import FiaGridDocument, FiaGridSnapshot
from f1_prediction.openf1 import LAP_COLUMNS, OpenF1Error, OpenF1Session
from f1_prediction.practice import PracticeFeatures


@dataclass
class FakeSession:
    laps: pd.DataFrame
    results: pd.DataFrame

    def __post_init__(self) -> None:
        self.load_calls: list[dict[str, object]] = []

    def load(self, **kwargs: object) -> None:
        self.load_calls.append(kwargs)


class FakeEvent(dict):
    def __init__(self, values: dict[str, object], sessions: dict[str, FakeSession]):
        super().__init__(values)
        self.sessions = sessions
        self.requested_sessions: list[str] = []

    def get_session(self, name: str) -> FakeSession:
        self.requested_sessions.append(name)
        return self.sessions[name]


def event_values(*, sprint: bool) -> dict[str, object]:
    if sprint:
        sessions = (
            "Practice 1",
            "Sprint Qualifying",
            "Sprint",
            "Qualifying",
            "Race",
        )
    else:
        sessions = (
            "Practice 1",
            "Practice 2",
            "Practice 3",
            "Qualifying",
            "Race",
        )
    values: dict[str, object] = {
        "EventName": "Example Grand Prix",
        "RoundNumber": 2,
    }
    values.update({f"Session{index}": name for index, name in enumerate(sessions, 1)})
    return values


def qualifying_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverId": "aaa",
                "Driver": "AAA",
                "Team": "One",
                "QualifyingPosition": 1,
                "QualifyingPositionPct": 0.0,
                "QualiTime_s": 90.0,
                "QualiGapPct": 0.0,
                "Q1GapPct": 0.0,
                "Q2GapPct": 0.0,
                "Q3GapPct": 0.0,
                "QualiGapToTeammatePct": pd.NA,
                "QualifyingPositionToTeammate": 0.0,
                "TeamQualifyingPositionMean": 1.0,
                "QualiStageReached": "Q3",
                "QualiTimeMissing": False,
                "QualifyingDataQuality": "USABLE",
            }
        ]
    )


def grid_snapshot() -> FiaGridSnapshot:
    grid = pd.DataFrame(
        [
            {
                "DriverId": "aaa",
                "GridPosition": 1,
                "StartsFromPitLane": False,
                "StartOrder": 1,
                "StartOrderPct": 0.0,
                "GridStatus": "FINAL",
                "GridSource": "https://fia.example/final.pdf",
                "GridPublishedAt": pd.Timestamp("2026-03-15T14:00:00+01:00"),
                "GridDocumentNumber": "40",
            }
        ]
    )
    document = FiaGridDocument(
        event_name="Example Grand Prix",
        year=2026,
        document_number="40",
        published_at=pd.Timestamp("2026-03-15T14:00:00+01:00"),
        documents_page_url="https://fia.example/documents",
        pdf_url="https://fia.example/final.pdf",
    )
    return FiaGridSnapshot(
        features=grid,
        document=document,
        pdf_sha256="abc123",
        pdf_path=Path("final.pdf"),
    )


def practice_features() -> PracticeFeatures:
    drivers = pd.DataFrame(
        [
            {
                "Driver": "AAA",
                "Team": "One",
                "PracticePaceDeltaPct": 0.0,
                "PracticeConsistencyPct": 0.1,
                "PracticeLongRunLapCount": 10,
                "PracticeStintCount": 1,
                "PracticeSessionsUsed": 1,
                "PracticeReferenceFallbackPct": 0.0,
                "PracticeDataReliability": 1.0,
                "PracticeDataQuality": "USABLE",
                "PracticeDataMissing": False,
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "Team": "One",
                "TeamPracticePaceDeltaPct": 0.0,
                "TeamPracticeDriverCount": 1,
                "TeamPracticeLongRunLapCount": 10,
                "TeamPracticeDataReliability": 1.0,
            }
        ]
    )
    return PracticeFeatures(drivers=drivers, teams=teams)


def sprint_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Driver": "AAA",
                "Team": "One",
                "SprintPaceDeltaPct": 0.0,
                "SprintConsistencyPct": 0.1,
                "SprintCleanLapCount": 15,
                "SprintDataReliability": 1.0,
                "SprintDataQuality": "USABLE",
                "SprintStartPosition": 1,
                "SprintFinishPosition": 1,
                "SprintStatus": "Finished",
                "SprintLapsCompleted": 20,
                "SprintPositionsGained": 0,
                "SprintLapsCompletedPct": 1.0,
                "SprintDNF": False,
                "TeamSprintPaceDeltaPct": 0.0,
            }
        ]
    )


def fake_sessions(values: dict[str, object]) -> dict[str, FakeSession]:
    return {
        str(name): FakeSession(
            pd.DataFrame({"session": [name]}),
            pd.DataFrame(
                {
                    "DriverId": ["aaa"],
                    "Abbreviation": ["AAA"],
                    "TeamName": ["One"],
                }
            ),
        )
        for key, name in values.items()
        if key.startswith("Session")
    }


def test_team_names_are_aligned_to_main_qualifying() -> None:
    laps = pd.DataFrame(
        {"Driver": ["AAA", "BBB", "RES"], "Team": ["", "Old", "Reserve"]}
    )
    qualifying = pd.DataFrame(
        {
            "Abbreviation": ["AAA", "BBB"],
            "TeamName": ["One", "Two"],
        }
    )

    aligned = _align_team_names(
        laps,
        qualifying,
        driver_column="Driver",
        team_column="Team",
    )

    assert aligned["Team"].tolist() == ["One", "Two", "Reserve"]

    race_drivers_only = _align_team_names(
        laps,
        qualifying,
        driver_column="Driver",
        team_column="Team",
        drop_unmatched=True,
    )
    assert race_drivers_only["Driver"].tolist() == ["AAA", "BBB"]
    assert race_drivers_only["Team"].tolist() == ["One", "Two"]


def patch_openf1(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """OpenF1'den istenen seans adlarını sırayla döndürmek için kaydeder."""
    requested: list[str] = []
    keys: dict[int, str] = {}

    def find(season: int, session_name: str, start_utc: object) -> OpenF1Session:
        key = len(keys) + 1
        keys[key] = session_name
        return OpenF1Session(key, 99, "Example Circuit")

    def laps(session_key: int) -> pd.DataFrame:
        requested.append(keys[session_key])
        return pd.DataFrame(columns=list(LAP_COLUMNS))

    monkeypatch.setattr("f1_prediction.collection.find_openf1_session", find)
    monkeypatch.setattr("f1_prediction.collection.fetch_openf1_laps", laps)
    return requested


def patch_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "f1_prediction.collection.build_practice_features",
        lambda *args, **kwargs: practice_features(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.build_qualifying_features",
        lambda *args, **kwargs: qualifying_features(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.build_sprint_features",
        lambda *args, **kwargs: sprint_features(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.fetch_fia_final_grid",
        lambda *args, **kwargs: grid_snapshot(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.build_track_geometry_features",
        lambda *args, **kwargs: {
            "TrackId": "99",
            "TrackName": "Example Circuit",
            "TrackLapDistanceKm": 5.0,
            "TrackLapDistanceApproximate": True,
            "TrackCornerCount": 10,
            "TrackCornersPerKm": 2.0,
            "TrackGeometryDataQuality": "USABLE",
            "TrackGeometryDataMissing": False,
            "TrackGeometrySource": "TEST",
        },
    )


def test_classification_ignores_sprint_qualifying() -> None:
    sessions = classify_event_sessions(event_values(sprint=True))

    assert sessions.practice == ("Practice 1",)
    assert sessions.qualifying == "Qualifying"
    assert sessions.sprint == "Sprint"


def test_conventional_weekend_loads_three_practices_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_builders(monkeypatch)
    lap_requests = patch_openf1(monkeypatch)
    values = event_values(sprint=False)
    event = FakeEvent(values, fake_sessions(values))
    event["Session5DateUtc"] = pd.Timestamp("2026-03-15T14:00:00Z")
    paths = WeekendDataPaths.from_project_root(tmp_path)

    result = collect_pre_race_weekend(
        2026,
        2,
        paths=paths,
        event_getter=lambda *_: event,
        cache_enabler=lambda *_: None,
    )

    assert lap_requests == ["Practice 1", "Practice 2", "Practice 3"]
    # FastF1'den yalnız Jolpica sonuçları ister; tur verisi F1 girişi gerektirir.
    assert event.requested_sessions == ["Qualifying"]
    assert event.sessions["Qualifying"].load_calls == [
        {"laps": False, "telemetry": False, "weather": False, "messages": False}
    ]
    assert not result.sessions.is_sprint_weekend
    assert result.feature_path is not None and result.feature_path.exists()
    assert result.features["DriverId"].tolist() == ["aaa"]
    assert result.features["TrackId"].tolist() == ["99"]
    assert result.features["TrackHistoryDataMissing"].all()
    assert result.race_start_utc == pd.Timestamp("2026-03-15T14:00:00Z")


def test_sprint_weekend_loads_sprint_but_never_sprint_qualifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_builders(monkeypatch)
    lap_requests = patch_openf1(monkeypatch)
    values = event_values(sprint=True)
    event = FakeEvent(values, fake_sessions(values))

    result = collect_pre_race_weekend(
        2026,
        2,
        paths=WeekendDataPaths.from_project_root(tmp_path),
        save=False,
        event_getter=lambda *_: event,
        cache_enabler=lambda *_: None,
    )

    assert lap_requests == ["Practice 1", "Sprint"]
    assert event.requested_sessions == ["Sprint", "Qualifying"]
    assert "Sprint Qualifying" not in lap_requests
    assert result.sessions.is_sprint_weekend
    assert result.features["SprintDataAvailable"].all()
    assert result.feature_path is None


def test_openf1_outage_marks_practice_missing_and_keeps_track_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Antrenman hesabı taklit edilmez: gerçek kod boş turlarla da çalışmalı.
    monkeypatch.setattr(
        "f1_prediction.collection.build_qualifying_features",
        lambda *args, **kwargs: qualifying_features(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.fetch_fia_final_grid",
        lambda *args, **kwargs: grid_snapshot(),
    )
    monkeypatch.setattr(
        "f1_prediction.collection.build_track_geometry_features",
        lambda track_id, track_name, *, year: {
            "TrackId": track_id,
            "TrackName": track_name,
            "TrackLapDistanceKm": None,
            "TrackLapDistanceApproximate": True,
            "TrackCornerCount": None,
            "TrackCornersPerKm": float("nan"),
            "TrackGeometryDataQuality": "MISSING",
            "TrackGeometryDataMissing": True,
            "TrackGeometrySource": "TEST",
        },
    )

    def unavailable(*args, **kwargs):
        raise OpenF1Error("erişilemiyor")

    monkeypatch.setattr("f1_prediction.collection.find_openf1_session", unavailable)
    paths = WeekendDataPaths.from_project_root(tmp_path)
    earlier = paths.processed / "track_history" / "2025"
    earlier.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Season": 2025,
                "RoundNumber": 5,
                "EventName": "Example Grand Prix",
                "TrackId": 99,
                "TrackName": "Example Circuit",
                "DriverId": "aaa",
                "StartOrder": 1,
                "ActualPosition": 1,
                "CleanMAEEligible": True,
            }
        ]
    ).to_csv(earlier / "round_05_track_rows.csv", index=False)
    values = event_values(sprint=False)

    result = collect_pre_race_weekend(
        2026,
        2,
        paths=paths,
        save=False,
        event_getter=lambda *_: FakeEvent(values, fake_sessions(values)),
        cache_enabler=lambda *_: None,
    )

    assert result.features["PracticeDataQuality"].eq("MISSING").all()
    assert result.features["PracticeDataMissing"].all()
    assert result.features["PreRaceDataMissing"].all()
    assert result.features["TrackId"].astype(str).eq("99").all()
    assert result.features["TrackName"].eq("Example Circuit").all()


def test_unknown_track_without_openf1_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_builders(monkeypatch)

    def unavailable(*args, **kwargs):
        raise OpenF1Error("erişilemiyor")

    monkeypatch.setattr("f1_prediction.collection.find_openf1_session", unavailable)
    values = event_values(sprint=False)

    with pytest.raises(WeekendCollectionError, match="pist kimliği"):
        collect_pre_race_weekend(
            2026,
            2,
            paths=WeekendDataPaths.from_project_root(tmp_path),
            save=False,
            event_getter=lambda *_: FakeEvent(values, fake_sessions(values)),
            cache_enabler=lambda *_: None,
        )


def test_missing_main_qualifying_is_rejected() -> None:
    values = event_values(sprint=False)
    values["Session4"] = "Something Else"

    with pytest.raises(WeekendCollectionError, match="ana Qualifying"):
        classify_event_sessions(values)
