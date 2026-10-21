"""Resmî yarış sonucunu model hedeflerine ve değerlendirme işaretlerine dönüştürür."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


class RaceResultError(ValueError):
    """Yarış sonucu eksik veya kendi içinde tutarsızsa oluşur."""


REQUIRED_RESULT_COLUMNS = {
    "DriverId",
    "Abbreviation",
    "TeamName",
    "Position",
    "Status",
    "Laps",
}
REQUIRED_GRID_COLUMNS = {"DriverId", "StartOrder"}
RACE_TARGET_COLUMNS = (
    "RaceOfficialPosition",
    "ActualPosition",
    "ActualPositionPct",
    "ActualPositionSource",
    "RaceStatus",
    "RaceStatusCategory",
    "RaceLapsCompleted",
    "RaceLapsCompletedPct",
    "RaceClassified",
    "RaceCompleted",
    "RaceDNF",
    "RaceDNS",
    "RaceDSQ",
    "CleanMAEEligible",
)


def _require_columns(
    frame: pd.DataFrame, required: set[str], label: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RaceResultError(f"{label} eksik sütunlar içeriyor: {', '.join(missing)}")


def _require_unique_driver(frame: pd.DataFrame, label: str) -> None:
    if frame["DriverId"].isna().any():
        raise RaceResultError(f"{label} içinde DriverId boş olamaz")
    duplicates = sorted(
        frame.loc[frame["DriverId"].duplicated(), "DriverId"].astype(str).unique()
    )
    if duplicates:
        raise RaceResultError(
            f"{label} içinde tekrarlanan pilot var: {', '.join(duplicates)}"
        )


def _status_flags(status: pd.Series) -> pd.DataFrame:
    normalized = status.fillna("").astype(str).str.strip().str.lower()
    completed = normalized.map(
        lambda value: value in {"finished", "lapped"}
        or bool(re.search(r"\+\s*\d+\s*lap", value))
    )
    dns = normalized.str.contains(
        r"did not start|not started|\bdns\b", regex=True
    )
    dsq = normalized.str.contains(
        r"disqualif|\bdsq\b|excluded", regex=True
    )
    dnf = ~completed & ~dns & ~dsq
    category = np.select(
        [completed, dns, dsq],
        ["FINISHED", "DNS", "DSQ"],
        default="DNF",
    )
    return pd.DataFrame(
        {
            "RaceCompleted": completed.astype(bool),
            "RaceDNF": dnf.astype(bool),
            "RaceDNS": dns.astype(bool),
            "RaceDSQ": dsq.astype(bool),
            "RaceStatusCategory": category,
        },
        index=status.index,
    )


def _complete_positions(
    official: pd.Series, start_order: pd.Series
) -> tuple[pd.Series, pd.Series]:
    driver_count = len(official)
    numeric = pd.to_numeric(official, errors="coerce")
    known = numeric.dropna()
    if ((known % 1) != 0).any():
        raise RaceResultError("Resmî yarış pozisyonları tam sayı olmalı")
    if ((known < 1) | (known > driver_count)).any():
        raise RaceResultError("Resmî yarış pozisyonu pilot sayısının dışında")
    if known.duplicated().any():
        raise RaceResultError("Resmî yarış sonucunda tekrarlanan pozisyon var")

    completed = numeric.astype("Float64")
    source = pd.Series("FIA", index=official.index, dtype="string")
    missing_indices = completed[completed.isna()].index.tolist()
    if missing_indices:
        start_numeric = pd.to_numeric(start_order, errors="coerce")
        if start_numeric.loc[missing_indices].isna().any():
            raise RaceResultError(
                "Pozisyonsuz pilotlar için final grid başlangıç sırası gerekli"
            )
        used = {int(value) for value in known}
        available = [position for position in range(1, driver_count + 1) if position not in used]
        ordered_missing = sorted(
            missing_indices,
            key=lambda index: float(start_numeric.loc[index]),
        )
        if len(available) != len(ordered_missing):
            raise RaceResultError("Eksik yarış pozisyonları güvenle tamamlanamadı")
        for index, position in zip(ordered_missing, available, strict=True):
            completed.loc[index] = position
            source.loc[index] = "GRID_FALLBACK"

    return completed.astype(int), source


def build_race_targets(
    results: pd.DataFrame,
    final_grid: pd.DataFrame,
    official_classification: pd.DataFrame,
) -> pd.DataFrame:
    """FIA sırasını korur; numarasız sonuçları final grid sırasıyla tamamlar."""
    _require_columns(results, REQUIRED_RESULT_COLUMNS, "Yarış sonuçları")
    _require_columns(final_grid, REQUIRED_GRID_COLUMNS, "Final grid")
    _require_columns(
        official_classification,
        {"DriverId", "RaceOfficialPosition"},
        "FIA final yarış sınıflandırması",
    )
    if results.empty:
        raise RaceResultError("Yarış sonuçları boş")
    _require_unique_driver(results, "Yarış sonuçları")
    _require_unique_driver(final_grid, "Final grid")
    _require_unique_driver(
        official_classification, "FIA final yarış sınıflandırması"
    )

    expected = set(final_grid["DriverId"])
    observed = set(results["DriverId"])
    if expected != observed:
        raise RaceResultError(
            "Yarış sonucu pilot listesi final grid ile uyuşmuyor. "
            f"Eksik: {sorted(expected - observed)}; Fazla: {sorted(observed - expected)}"
        )
    official_ids = set(official_classification["DriverId"])
    if expected != official_ids:
        raise RaceResultError(
            "FIA yarış pilot listesi final grid ile uyuşmuyor. "
            f"Eksik: {sorted(expected - official_ids)}; "
            f"Fazla: {sorted(official_ids - expected)}"
        )

    target = results[
        ["DriverId", "Abbreviation", "TeamName", "Status", "Laps"]
    ].copy()
    target = target.rename(
        columns={
            "Abbreviation": "RaceDriver",
            "TeamName": "RaceTeam",
            "Status": "RaceStatus",
            "Laps": "RaceLapsCompleted",
        }
    )
    target = target.merge(
        final_grid[["DriverId", "StartOrder"]],
        on="DriverId",
        how="left",
        validate="one_to_one",
    )
    target = target.merge(
        official_classification[["DriverId", "RaceOfficialPosition"]],
        on="DriverId",
        how="left",
        validate="one_to_one",
    )
    target["RaceClassified"] = pd.to_numeric(
        target["RaceOfficialPosition"], errors="coerce"
    ).notna()
    target["ActualPosition"], target["ActualPositionSource"] = _complete_positions(
        target["RaceOfficialPosition"], target["StartOrder"]
    )
    driver_count = len(target)
    target["ActualPositionPct"] = np.where(
        driver_count > 1,
        (target["ActualPosition"] - 1) / (driver_count - 1),
        np.nan,
    )
    target["RaceOfficialPosition"] = pd.to_numeric(
        target["RaceOfficialPosition"], errors="coerce"
    )
    target["RaceLapsCompleted"] = pd.to_numeric(
        target["RaceLapsCompleted"], errors="coerce"
    )
    max_laps = target["RaceLapsCompleted"].max(skipna=True)
    target["RaceLapsCompletedPct"] = np.where(
        pd.notna(max_laps) and max_laps > 0,
        target["RaceLapsCompleted"] / max_laps,
        np.nan,
    )
    target = pd.concat([target, _status_flags(target["RaceStatus"])], axis=1)
    target["CleanMAEEligible"] = target["RaceCompleted"]
    return target.drop(columns="StartOrder").sort_values("ActualPosition").reset_index(
        drop=True
    )
