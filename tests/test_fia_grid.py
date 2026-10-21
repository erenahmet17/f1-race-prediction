from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.fia_grid import (
    attach_driver_ids,
    parse_final_grid_links,
    parse_grid_page_words,
    parse_penalty_notes,
)
from f1_prediction.grid import GridDataError


def word(text: str, x0: float, x1: float, top: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x1, "top": top}


def test_only_exact_final_race_grid_is_discovered() -> None:
    html = """
    <a href="/provisional.pdf">Doc 50 - Provisional Starting Grid Published on 05.09.26 20:00 CET</a>
    <a href="/sprint.pdf">Doc 51 - Final Sprint Starting Grid Published on 06.09.26 11:00 CET</a>
    <a href="/final.pdf"><span>Doc 59 - Final Starting Grid</span> Published on 06.09.26 14:00 CET</a>
    <a href="/old.pdf">Doc 37 - Final Starting Grid Published on 07.09.25 14:00 CET</a>
    """

    documents = parse_final_grid_links(
        html,
        event_name="Italian Grand Prix",
        year=2026,
        documents_page_url="https://www.fia.com/documents/event/Italian%20Grand%20Prix",
    )

    assert len(documents) == 1
    assert documents[0].document_number == "59"
    assert documents[0].pdf_url == "https://www.fia.com/final.pdf"


def test_two_column_grid_and_pit_lane_are_parsed_by_coordinates() -> None:
    words = [
        word("1", 52, 59, 190),
        word("10", 70, 79, 188),
        word("Pierre", 85, 110, 188),
        word("2", 317, 324, 202),
        word("63", 335, 344, 200),
        word("George", 350, 380, 200),
        word("3", 52, 59, 220),
        word("16", 70, 79, 218),
        word("Charles", 85, 115, 218),
        word("DRIVERS", 63, 94, 500),
        word("REQUIRED", 96, 133, 500),
        word("PIT", 210, 220, 500),
        word("14", 70, 79, 515),
        word("Fernando", 85, 120, 515),
        word("PENALTIES", 276, 315, 588),
        word("12", 34, 43, 600),
    ]

    grid = parse_grid_page_words(words, page_width=595).set_index("CarNumber")

    assert grid.loc["10", "GridPosition"] == 1
    assert grid.loc["63", "GridPosition"] == 2
    assert grid.loc["16", "GridPosition"] == 3
    assert bool(grid.loc["14", "StartsFromPitLane"])
    assert "12" not in grid.index


def test_non_contiguous_grid_is_rejected() -> None:
    words = [
        word("1", 52, 59, 190),
        word("10", 70, 79, 188),
        word("3", 52, 59, 220),
        word("16", 70, 79, 218),
        word("PENALTIES", 276, 315, 588),
    ]

    with pytest.raises(GridDataError, match="kesintisiz"):
        parse_grid_page_words(words, page_width=595)


def test_car_numbers_are_mapped_to_driver_ids() -> None:
    grid = pd.DataFrame(
        [
            {"CarNumber": "10", "GridPosition": 1, "StartsFromPitLane": False},
            {"CarNumber": "63", "GridPosition": 2, "StartsFromPitLane": False},
        ]
    )
    qualifying = pd.DataFrame(
        [
            {"DriverNumber": "10", "DriverId": "pierre_gasly"},
            {"DriverNumber": "63", "DriverId": "george_russell"},
        ]
    )

    mapped = attach_driver_ids(grid, qualifying).set_index("CarNumber")

    assert mapped.loc["10", "DriverId"] == "pierre_gasly"
    assert mapped.loc["63", "DriverId"] == "george_russell"


def test_fia_penalty_notes_are_classified_by_car() -> None:
    text = """
    * PENALTIES
    Car 12 - 30 place grid penalty - Additional power unit elements have been used - Stewards' document no. 19
    Car 81 - 3 place grid penalty - Impeding another driver - Stewards' document no. 47
    Car 14 - Required to start from the pit lane - Car modified whilst under Parc Fermé conditions
    and additional power unit elements have been used - Stewards' document no. 56
    Cars 87 & 18 - Permitted to start - Stewards' document no. 55
    Garry Connelly Mathieu Remmerie The Stewards
    """

    notes = parse_penalty_notes(text).set_index("CarNumber")

    assert notes.loc["12", "GridPenaltyPositions"] == 30
    assert bool(notes.loc["12", "GridPenaltyPowerUnit"])
    assert bool(notes.loc["81", "GridPenaltySporting"])
    assert bool(notes.loc["14", "GridPenaltyPitLane"])
    assert bool(notes.loc["14", "GridPenaltyParcFerme"])
    assert notes.loc["14", "GridPenaltyReason"] == "MULTIPLE"
    assert bool(notes.loc["87", "GridPermissionToStart"])
    assert not bool(notes.loc["87", "GridPenaltyApplied"])
