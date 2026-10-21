from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.grid import (
    FinalGridNotReadyError,
    GridDataError,
    GridDocumentMetadata,
    build_final_grid_features,
)


def metadata(document_type: str = "FINAL_STARTING_GRID") -> GridDocumentMetadata:
    return GridDocumentMetadata(
        document_type=document_type,
        source="https://www.fia.com/example-final-grid.pdf",
        published_at=pd.Timestamp("2026-09-13T14:00:00+02:00"),
        document_number="54",
    )


def grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"DriverId": "aaa", "GridPosition": 1, "StartsFromPitLane": False},
            {"DriverId": "bbb", "GridPosition": 3, "StartsFromPitLane": False},
            {"DriverId": "ccc", "GridPosition": None, "StartsFromPitLane": True},
        ]
    )


def test_provisional_grid_blocks_prediction() -> None:
    with pytest.raises(FinalGridNotReadyError, match="tahmin oluşturulamaz"):
        build_final_grid_features(grid(), metadata("PROVISIONAL_STARTING_GRID"))


def test_final_grid_creates_contiguous_start_order() -> None:
    features = build_final_grid_features(
        grid(), metadata(), expected_driver_ids={"aaa", "bbb", "ccc"}
    ).set_index("DriverId")

    assert features.loc["aaa", "StartOrder"] == 1
    assert features.loc["bbb", "StartOrder"] == 2
    assert features.loc["ccc", "StartOrder"] == 3
    assert bool(features.loc["ccc", "StartsFromPitLane"])
    assert features.loc["aaa", "GridStatus"] == "FINAL"


def test_grid_driver_list_must_match_qualifying() -> None:
    with pytest.raises(GridDataError, match="qualifying ile uyuşmuyor"):
        build_final_grid_features(
            grid(), metadata(), expected_driver_ids={"aaa", "bbb", "ddd"}
        )


def test_duplicate_grid_position_is_rejected() -> None:
    invalid = grid()
    invalid.loc[1, "GridPosition"] = 1

    with pytest.raises(GridDataError, match="tekrarlanan GridPosition"):
        build_final_grid_features(invalid, metadata())
