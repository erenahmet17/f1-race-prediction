from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1_prediction.track import (
    attach_track_features,
    build_track_geometry_features,
    build_track_history_features,
    load_available_track_history,
    track_identity_from_history,
)


def fake_circuit(track_id: str, year: int) -> dict[str, object]:
    # 1250 metre kenarlı kare: çevresi 5 km, koordinat birimi metrenin onda biri.
    return {
        "x": [0, 12500, 12500, 0],
        "y": [0, 0, 12500, 12500],
        "corners": [{"number": number, "letter": ""} for number in range(1, 11)],
    }


def geometry_features() -> dict[str, object]:
    return build_track_geometry_features(
        "99", "Example Circuit", year=2026, circuit_getter=fake_circuit
    )


def history() -> pd.DataFrame:
    rows = []
    events = [
        (2025, 1, "99", [(1, 2), (2, 1), (3, 3)]),
        (2025, 2, "22", [(1, 1), (2, 2), (3, 3)]),
        (2026, 1, "22", [(1, 3), (2, 2), (3, 1)]),
        (2026, 2, "99", [(1, 3), (2, 1), (3, 2)]),
        (2026, 5, "99", [(1, 2), (2, 3), (3, 1)]),
    ]
    for season, round_number, track_id, positions in events:
        for index, (start, finish) in enumerate(positions):
            rows.append(
                {
                    "Season": season,
                    "RoundNumber": round_number,
                    "TrackId": track_id,
                    "DriverId": f"d{index}",
                    "StartOrder": start,
                    "ActualPosition": finish,
                    "CleanMAEEligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_geometry_uses_circuit_outline_distance_and_corner_markers() -> None:
    features = geometry_features()

    assert features["TrackId"] == "99"
    assert features["TrackName"] == "Example Circuit"
    assert features["TrackLapDistanceKm"] == 5.0
    assert features["TrackLapDistanceApproximate"]
    assert features["TrackCornerCount"] == 10
    assert features["TrackCornersPerKm"] == 2.0
    assert features["TrackGeometryDataQuality"] == "USABLE"


def test_geometry_is_marked_missing_when_circuit_is_unavailable() -> None:
    features = build_track_geometry_features(
        "99", "Example Circuit", year=2026, circuit_getter=lambda *_: None
    )

    assert features["TrackId"] == "99"
    assert features["TrackLapDistanceKm"] is None
    assert features["TrackCornerCount"] is None
    assert features["TrackGeometryDataQuality"] == "MISSING"
    assert features["TrackGeometryDataMissing"]


def test_implausible_circuit_length_is_ignored() -> None:
    def tiny_circuit(track_id: str, year: int) -> dict[str, object]:
        return {**fake_circuit(track_id, year), "x": [0, 1], "y": [0, 1]}

    features = build_track_geometry_features(
        "99", "Example Circuit", year=2026, circuit_getter=tiny_circuit
    )

    assert features["TrackLapDistanceKm"] is None
    assert features["TrackCornerCount"] == 10
    assert features["TrackGeometryDataQuality"] == "PARTIAL"


def test_track_identity_comes_from_latest_record_of_same_event() -> None:
    rows = pd.DataFrame(
        {
            "EventName": ["Example Grand Prix", "Other Grand Prix", "Example Grand Prix"],
            "TrackId": [98, 22, 99],
            "TrackName": ["Old Circuit", "Other", "Example Circuit"],
        }
    )

    assert track_identity_from_history(rows, "Example Grand Prix") == (
        "99",
        "Example Circuit",
    )
    assert track_identity_from_history(rows, "New Grand Prix") is None
    assert track_identity_from_history(pd.DataFrame(), "Example Grand Prix") is None


def test_track_history_excludes_target_and_future_rounds() -> None:
    base = history()
    profile = build_track_history_features(
        base,
        track_id="99",
        target_season=2026,
        target_round=2,
    )
    changed_future = base.copy()
    changed_future.loc[
        (changed_future["Season"] == 2026)
        & (changed_future["RoundNumber"] >= 2),
        "ActualPosition",
    ] = [3, 2, 1, 3, 1, 2]
    second = build_track_history_features(
        changed_future,
        track_id="99",
        target_season=2026,
        target_round=2,
    )

    assert profile == second
    assert profile["TrackHistoryRaceCount"] == 1
    assert not profile["TrackHistoryDataMissing"]


def test_dnf_rows_do_not_change_track_movement_profile() -> None:
    base = history()
    dnf = base.iloc[[0]].copy()
    dnf["DriverId"] = "dnf"
    dnf["StartOrder"] = 1
    dnf["ActualPosition"] = 20
    dnf["CleanMAEEligible"] = False
    first = pd.concat([base, dnf], ignore_index=True)
    changed_dnf = dnf.copy()
    changed_dnf["StartOrder"] = 20
    changed_dnf["ActualPosition"] = 1
    second = pd.concat([base, changed_dnf], ignore_index=True)

    original = build_track_history_features(
        first, track_id="99", target_season=2026, target_round=2
    )
    with_dnf = build_track_history_features(
        second,
        track_id="99",
        target_season=2026,
        target_round=2,
    )

    assert original == with_dnf


def test_unknown_track_returns_explicit_missing_profile() -> None:
    profile = build_track_history_features(
        history(), track_id="404", target_season=2026, target_round=4
    )

    assert profile["TrackHistoryDataMissing"]
    assert profile["TrackHistoryRaceCount"] == 0
    assert np.isnan(profile["TrackHistoricalPositionChangeRelative"])


def test_attach_track_features_repeats_only_event_level_values() -> None:
    weekend = pd.DataFrame({"DriverId": ["aaa", "bbb"]})
    geometry = geometry_features()

    result = attach_track_features(
        weekend,
        geometry,
        history(),
        target_season=2026,
        target_round=2,
    )

    assert result["TrackId"].eq("99").all()
    assert result["TrackHistoryRaceCount"].eq(1).all()


def test_disk_loader_never_reads_current_or_future_rounds(tmp_path: Path) -> None:
    root = tmp_path / "history" / "2026"
    root.mkdir(parents=True)
    for round_number in (1, 2, 3):
        pd.DataFrame(
            [
                {
                    "Season": 2026,
                    "RoundNumber": round_number,
                    "TrackId": "99",
                }
            ]
        ).to_csv(root / f"round_{round_number:02d}_training_rows.csv", index=False)

    loaded = load_available_track_history(
        tmp_path, target_season=2026, target_round=3
    )

    assert loaded["RoundNumber"].tolist() == [1, 2]
