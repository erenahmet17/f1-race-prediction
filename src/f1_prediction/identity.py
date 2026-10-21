"""FastF1 sonuçlarındaki pilot kimliklerini güvenli biçimde tamamlar."""

from __future__ import annotations

import pandas as pd


class DriverIdentityError(ValueError):
    """Pilot kimliği güvenle üretilemediğinde oluşur."""


_INVALID_TEXT = {"", "nan", "none", "<na>", "nat", "null"}


def normalize_fastf1_driver_ids(results: pd.DataFrame) -> pd.DataFrame:
    """Seanslar arasında değişebilen DriverId yerine üç harfli kodu sabitler."""
    required = {"DriverId", "Abbreviation"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise DriverIdentityError(
            f"Pilot kimliği için eksik sütunlar var: {', '.join(missing)}"
        )

    normalized = results.copy()
    abbreviations = normalized["Abbreviation"].astype("string").str.strip().str.upper()
    invalid_abbreviations = (
        abbreviations.isna() | abbreviations.str.lower().isin(_INVALID_TEXT)
    )
    if invalid_abbreviations.any():
        raise DriverIdentityError(
            "Abbreviation boş; seanslar arası sabit pilot kimliği üretilemedi"
        )

    driver_ids = "abbr:" + abbreviations
    if driver_ids.duplicated().any():
        duplicates = sorted(driver_ids[driver_ids.duplicated()].astype(str).unique())
        raise DriverIdentityError(
            f"Normalize edilmiş pilot kimliği tekrarlanıyor: {', '.join(duplicates)}"
        )
    normalized["DriverId"] = driver_ids
    return normalized
