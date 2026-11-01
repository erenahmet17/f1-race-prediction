from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import f1_prediction.history as history_module
from f1_prediction.collection import WeekendDataPaths, weekend_features_path
from f1_prediction.history import (
    HistoryDataError,
    InsufficientHistoryError,
    build_completed_round_rows,
    collect_completed_round,
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


class FakeEvent(dict):
    def get_session(self, name: str) -> SimpleNamespace:
        results = pd.DataFrame(
            {
                "DriverId": ["max_verstappen", "leclerc"],
                "DriverNumber": ["3", "16"],
                "Abbreviation": ["VER", "LEC"],
                "TeamName": ["Red Bull", "Ferrari"],
                "Position": [2, 1],
                "Status": ["Finished", "Finished"],
                "Laps": [56, 56],
            }
        )
        return SimpleNamespace(load=lambda **kwargs: None, results=results)


def fake_event(season: int, round_number: int) -> FakeEvent:
    return FakeEvent(
        EventName="Test Grand Prix",
        Session1="Practice 1",
        Session4="Qualifying",
        Session5="Race",
    )


def save_pre_race_table(paths: WeekendDataPaths) -> Path:
    path = weekend_features_path(paths, 2026, 2)
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "Season": [2026, 2026],
            "RoundNumber": [2, 2],
            "DriverId": ["abbr:VER", "abbr:LEC"],
            "Driver": ["VER", "LEC"],
            "Team": ["Red Bull Racing", "Ferrari"],
            "StartOrder": [1, 2],
        }
    ).to_csv(path, index=False)
    return path


def patch_fia(monkeypatch, *, grid: list[int]) -> None:
    ids = ["abbr:VER", "abbr:LEC"]
    monkeypatch.setattr(
        history_module,
        "fetch_fia_final_grid",
        lambda *args, **kwargs: SimpleNamespace(
            features=pd.DataFrame({"DriverId": ids, "StartOrder": grid})
        ),
    )
    monkeypatch.setattr(
        history_module,
        "fetch_fia_final_race_classification",
        lambda *args, **kwargs: SimpleNamespace(
            classification=pd.DataFrame(
                {"DriverId": ids, "RaceOfficialPosition": [2, 1]}
            )
        ),
    )


def forbid_recollection(*args, **kwargs):
    raise AssertionError("Kayıtlı tablo varken yarış öncesi veri yeniden toplanmamalı")


def test_result_reuses_saved_pre_race_table_without_recollecting(
    tmp_path: Path, monkeypatch
) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    saved = save_pre_race_table(paths)
    before = saved.read_bytes()
    patch_fia(monkeypatch, grid=[1, 2])
    monkeypatch.setattr(history_module, "collect_pre_race_weekend", forbid_recollection)

    result = collect_completed_round(
        2026,
        2,
        paths=paths,
        event_getter=fake_event,
        cache_enabler=lambda path: None,
        reuse_saved_pre_race=True,
    )

    assert saved.read_bytes() == before
    assert result.history_path is not None and result.history_path.exists()
    assert result.pre_race.event_name == "Test Grand Prix"
    rows = pd.read_csv(result.history_path).set_index("Driver")
    assert rows.loc["VER", "ActualPosition"] == 2
    assert rows.loc["LEC", "ActualPosition"] == 1
    assert rows.loc["VER", "Team"] == "Red Bull Racing"


def test_saved_pre_race_table_must_match_fia_final_grid(
    tmp_path: Path, monkeypatch
) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    save_pre_race_table(paths)
    patch_fia(monkeypatch, grid=[2, 1])
    monkeypatch.setattr(history_module, "collect_pre_race_weekend", forbid_recollection)

    with pytest.raises(HistoryDataError, match="FIA final gridiyle"):
        collect_completed_round(
            2026,
            2,
            paths=paths,
            event_getter=fake_event,
            cache_enabler=lambda path: None,
            reuse_saved_pre_race=True,
        )

    assert not (paths.processed / "history" / "2026" / "round_02_training_rows.csv").exists()


def test_result_recollects_by_default_even_when_table_is_saved(
    tmp_path: Path, monkeypatch
) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    save_pre_race_table(paths)

    def recollect(*args, **kwargs):
        raise RuntimeError("yeniden toplandı")

    monkeypatch.setattr(history_module, "collect_pre_race_weekend", recollect)

    with pytest.raises(RuntimeError, match="yeniden toplandı"):
        collect_completed_round(2026, 2, paths=paths)


def test_first_round_prediction_is_explicitly_blocked() -> None:
    with pytest.raises(InsufficientHistoryError, match="ilk yarışı"):
        select_history_before_round(
            pd.DataFrame(), season=2026, prediction_round=1
        )
