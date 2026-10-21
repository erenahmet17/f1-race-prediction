from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from f1_prediction.backfill import (
    BackfillError,
    backfill_track_history,
    backfill_training_history,
    build_compact_track_history_rows,
    collect_compact_track_round,
)
from f1_prediction.collection import WeekendDataPaths
from f1_prediction.track import load_available_track_history


def race_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverId": "aaa",
                "GridPosition": 1,
                "Position": 2,
                "Status": "Finished",
            },
            {
                "DriverId": "bbb",
                "GridPosition": 2,
                "Position": 1,
                "Status": "Lapped",
            },
            {
                "DriverId": "ccc",
                "GridPosition": 3,
                "Position": 3,
                "Status": "Retired",
            },
            {
                "DriverId": "ddd",
                "GridPosition": 0,
                "Position": 4,
                "Status": "Finished",
            },
        ]
    )


def compact_rows(season: int, round_number: int) -> pd.DataFrame:
    return build_compact_track_history_rows(
        race_results(),
        season=season,
        round_number=round_number,
        event_name="Example Grand Prix",
        track_id="99",
        track_name="Example Circuit",
    )


def test_compact_rows_use_only_completed_valid_grid_starters() -> None:
    rows = compact_rows(2025, 1).set_index("DriverId")

    assert bool(rows.loc["aaa", "CleanMAEEligible"])
    assert bool(rows.loc["bbb", "CleanMAEEligible"])
    assert not bool(rows.loc["ccc", "CleanMAEEligible"])
    assert not bool(rows.loc["ddd", "CleanMAEEligible"])
    assert pd.isna(rows.loc["ddd", "StartOrder"])


class FakeSession:
    def __init__(self) -> None:
        self.results = race_results()
        self.event = {"EventName": "Example Grand Prix"}
        self.session_info = {
            "Meeting": {"Circuit": {"Key": 99, "ShortName": "Example Circuit"}}
        }
        self.load_kwargs: dict[str, object] | None = None

    def load(self, **kwargs: object) -> None:
        self.load_kwargs = kwargs


def test_single_track_round_is_written_atomically(tmp_path: Path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    session = FakeSession()

    output = collect_compact_track_round(
        2025,
        1,
        paths=paths,
        session_getter=lambda *_: session,
        cache_enabler=lambda *_: None,
    )

    assert output.exists()
    assert not output.with_suffix(".csv.tmp").exists()
    assert pd.read_csv(output)["TrackId"].astype(str).eq("99").all()
    assert session.load_kwargs == {
        "laps": False,
        "telemetry": False,
        "weather": False,
        "messages": False,
    }


def test_single_track_round_falls_back_when_fastf1_results_are_invalid(
    tmp_path: Path,
) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    session = FakeSession()
    session.results["DriverId"] = pd.NA
    fallback_calls: list[tuple[int, int]] = []

    def fallback(season: int, round_number: int) -> pd.DataFrame:
        fallback_calls.append((season, round_number))
        return race_results()

    output = collect_compact_track_round(
        2025,
        24,
        paths=paths,
        session_getter=lambda *_: session,
        cache_enabler=lambda *_: None,
        fallback_results_getter=fallback,
    )

    rows = pd.read_csv(output)
    assert fallback_calls == [(2025, 24)]
    assert rows["TrackHistorySource"].eq("JOLPICA_RESULTS_FALLBACK").all()


def test_track_backfill_resumes_after_valid_existing_file(tmp_path: Path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    existing = (
        paths.processed / "track_history" / "2025" / "round_01_track_rows.csv"
    )
    existing.parent.mkdir(parents=True)
    compact_rows(2025, 1).to_csv(existing, index=False)
    called: list[int] = []

    def collector(season: int, round_number: int, *, paths: WeekendDataPaths) -> Path:
        called.append(round_number)
        destination = (
            paths.processed
            / "track_history"
            / str(season)
            / f"round_{round_number:02d}_track_rows.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        compact_rows(season, round_number).to_csv(destination, index=False)
        return destination

    summaries = backfill_track_history(
        2025,
        2025,
        paths=paths,
        schedule_getter=lambda *args, **kwargs: pd.DataFrame(
            {"RoundNumber": [1, 2, 3]}
        ),
        collector=collector,
    )

    assert called == [2, 3]
    assert summaries[2025].skipped_rounds == (1,)
    assert summaries[2025].collected_rounds == (2, 3)


def test_invalid_existing_track_file_stops_without_overwrite(tmp_path: Path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    existing = (
        paths.processed / "track_history" / "2025" / "round_01_track_rows.csv"
    )
    existing.parent.mkdir(parents=True)
    existing.write_text("bad,data\n1,2\n", encoding="utf-8")

    with pytest.raises(BackfillError, match="geçersiz"):
        backfill_track_history(
            2025,
            2025,
            paths=paths,
            schedule_getter=lambda *args, **kwargs: pd.DataFrame(
                {"RoundNumber": [1]}
            ),
        )

    assert existing.read_text(encoding="utf-8") == "bad,data\n1,2\n"


def test_training_backfill_skips_valid_and_collects_missing(tmp_path: Path) -> None:
    paths = WeekendDataPaths.from_project_root(tmp_path)
    existing = (
        paths.processed / "history" / "2026" / "round_01_training_rows.csv"
    )
    existing.parent.mkdir(parents=True)
    base = compact_rows(2026, 1)
    base.to_csv(existing, index=False)
    called: list[int] = []

    def collector(
        season: int,
        round_number: int,
        *,
        paths: WeekendDataPaths,
        save: bool,
    ) -> SimpleNamespace:
        called.append(round_number)
        destination = (
            paths.processed
            / "history"
            / str(season)
            / f"round_{round_number:02d}_training_rows.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        compact_rows(season, round_number).to_csv(destination, index=False)
        return SimpleNamespace(history_path=destination)

    summary = backfill_training_history(
        2026,
        3,
        paths=paths,
        collector=collector,
    )

    assert called == [2, 3]
    assert summary.skipped_rounds == (1,)
    assert summary.collected_rounds == (2, 3)


def test_full_training_history_is_preferred_over_compact_duplicate(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    compact_dir = processed / "track_history" / "2025"
    full_dir = processed / "history" / "2025"
    compact_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)

    compact = compact_rows(2025, 1)
    compact["TrackId"] = "compact"
    compact.to_csv(compact_dir / "round_01_track_rows.csv", index=False)
    full = compact_rows(2025, 1)
    full["TrackId"] = "full"
    full.to_csv(full_dir / "round_01_training_rows.csv", index=False)
    compact_rows(2025, 2).to_csv(
        compact_dir / "round_02_track_rows.csv", index=False
    )

    loaded = load_available_track_history(
        processed,
        target_season=2026,
        target_round=1,
    )

    round_one = loaded[loaded["RoundNumber"] == 1]
    assert round_one["TrackId"].astype(str).eq("full").all()
    assert sorted(loaded["RoundNumber"].unique()) == [1, 2]


def test_through_round_requires_single_season(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tek sezon"):
        backfill_track_history(
            2024,
            2025,
            paths=WeekendDataPaths.from_project_root(tmp_path),
            through_round=10,
        )
