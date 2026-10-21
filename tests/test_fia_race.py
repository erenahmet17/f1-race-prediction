from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.fia_race import (
    _attach_driver_ids,
    parse_classification_page_words,
    parse_final_race_links,
)
from f1_prediction.race import RaceResultError


def word(text: str, x0: float, top: float, width: float = 8.0) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top}


def test_link_parser_accepts_only_final_race_document() -> None:
    html = """
    <a href="/final.pdf">Doc 70 - Final Race Classification Published on 06.09.26 20:25 CET</a>
    <a href="/provisional.pdf">Doc 66 - Provisional Race Classification Published on 06.09.26 17:25 CET</a>
    <a href="/qualifying.pdf">Doc 50 - Final Qualifying Classification Published on 05.09.26 20:42 CET</a>
    """

    documents = parse_final_race_links(
        html,
        event_name="Italian Grand Prix",
        year=2026,
        documents_page_url="https://www.fia.com/documents/event/Italian%20Grand%20Prix",
    )

    assert len(documents) == 1
    assert documents[0].document_number == "70"
    assert documents[0].pdf_url == "https://www.fia.com/final.pdf"


def test_coordinate_parser_reads_classified_and_not_classified_cars() -> None:
    words = [
        word("NO", 41, 180),
        word("1", 31, 193),
        word("12", 45, 193),
        word("2", 31, 205),
        word("63", 45, 205),
        word("NOT", 274, 430),
        word("CLASSIFIED", 291, 430, 40),
        word("18", 45, 445),
        word("14", 45, 457),
        word("FASTEST", 277, 494, 35),
    ]

    parsed = parse_classification_page_words(words, 595).astype(
        {"CarNumber": "string"}
    )

    assert parsed["CarNumber"].tolist() == ["12", "63", "18", "14"]
    assert parsed["RaceOfficialPosition"].iloc[:2].tolist() == [1, 2]
    assert parsed["RaceOfficialPosition"].iloc[2:].isna().all()


def test_coordinate_parser_rejects_non_contiguous_positions() -> None:
    words = [
        word("NO", 41, 180),
        word("1", 31, 193),
        word("12", 45, 193),
        word("3", 31, 205),
        word("63", 45, 205),
        word("NOT", 274, 430),
        word("CLASSIFIED", 291, 430, 40),
        word("FASTEST", 277, 494, 35),
    ]

    with pytest.raises(RaceResultError, match="kesintisiz"):
        parse_classification_page_words(words, 595)


def test_car_numbers_map_to_fastf1_driver_ids() -> None:
    classification = pd.DataFrame(
        [
            {"CarNumber": "12", "RaceOfficialPosition": 1},
            {"CarNumber": "63", "RaceOfficialPosition": 2},
        ]
    )
    results = pd.DataFrame(
        [
            {"DriverId": "antonelli", "DriverNumber": "12"},
            {"DriverId": "russell", "DriverNumber": "63"},
        ]
    )

    mapped = _attach_driver_ids(classification, results)

    assert mapped["DriverId"].tolist() == ["antonelli", "russell"]
