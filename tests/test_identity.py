from __future__ import annotations

import pandas as pd
import pytest

from f1_prediction.identity import DriverIdentityError, normalize_fastf1_driver_ids


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
