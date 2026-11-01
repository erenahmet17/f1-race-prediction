from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.identity import (
    DriverIdentityError,
    normalize_fastf1_driver_ids,
    normalize_team_names,
)


def test_all_fastf1_driver_ids_use_stable_prefixed_abbreviation() -> None:
    results = pd.DataFrame(
        {
            "DriverId": ["russell", "nan", pd.NA],
            "Abbreviation": ["RUS", "LIN", "COL"],
        }
    )

    normalized = normalize_fastf1_driver_ids(results)

    assert normalized["DriverId"].tolist() == [
        "abbr:RUS",
        "abbr:LIN",
        "abbr:COL",
    ]


def test_driver_identity_requires_at_least_one_usable_identifier() -> None:
    results = pd.DataFrame(
        {"DriverId": [pd.NA], "Abbreviation": [pd.NA]}
    )

    with pytest.raises(DriverIdentityError, match="Abbreviation boş"):
        normalize_fastf1_driver_ids(results)


def test_missing_abbreviation_comes_from_surname() -> None:
    results = pd.DataFrame(
        {
            "DriverId": ["ryo_hirakawa", "hamilton"],
            "Abbreviation": ["nan", "HAM"],
            "LastName": ["Hirakawa", "Hamilton"],
        }
    )

    normalized = normalize_fastf1_driver_ids(results)

    assert normalized["Abbreviation"].tolist() == ["HIR", "HAM"]
    assert normalized["DriverId"].tolist() == ["abbr:HIR", "abbr:HAM"]


def test_missing_abbreviation_without_surname_is_rejected() -> None:
    results = pd.DataFrame(
        {"DriverId": ["x"], "Abbreviation": ["nan"], "LastName": [pd.NA]}
    )

    with pytest.raises(DriverIdentityError, match="Abbreviation boş"):
        normalize_fastf1_driver_ids(results)


def test_jolpica_team_names_match_f1_api_names_and_others_stay() -> None:
    results = pd.DataFrame(
        {"TeamName": ["Red Bull", "RB F1 Team", "Alpine F1 Team", "Cadillac F1 Team", "Ferrari"]}
    )

    assert normalize_team_names(results)["TeamName"].tolist() == [
        "Red Bull Racing",
        "Racing Bulls",
        "Alpine",
        "Cadillac",
        "Ferrari",
    ]
