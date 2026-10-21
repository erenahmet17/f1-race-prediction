"""Streamlit arayüzü için kayıtlı yarış ve tahmin verilerini okur."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RaceView:
    rows: pd.DataFrame
    source: str | None
    event_name: str
    mae: float | None
    clean_mae: float | None


# Koyu tonlar, tablo içindeki açık renkli yazıyla okunabilir kalır.
TEAM_ROW_COLORS = {
    "mclaren": "#754013",
    "ferrari": "#75242c",
    "mercedes": "#175a55",
    "red bull racing": "#283e76",
    "red bull": "#283e76",
    "williams": "#254f85",
    "aston martin": "#205a45",
    "alpine": "#364e83",
    "audi": "#5b303b",
    "haas f1 team": "#535965",
    "haas": "#535965",
    "racing bulls": "#324f7a",
    "cadillac": "#4d5260",
}


def style_team_rows(frame: pd.DataFrame, *, team_column: str = "Team"):
    """Pilot satırlarını takım rengine göre boyar; tanınmayan takımı nötr bırakır."""
    if team_column not in frame:
        return frame.style

    def color_row(row: pd.Series) -> list[str]:
        team = str(row[team_column]).strip().casefold()
        background = TEAM_ROW_COLORS.get(team)
        if background is None:
            return [""] * len(row)
        return [f"background-color: {background}; color: #f8f9fb"] * len(row)

    return frame.style.apply(color_row, axis=1)


def available_seasons(processed: Path) -> list[int]:
    """Geçmiş, backtest veya canlı tahmin kaydı bulunan sezonları listeler."""
    seasons: set[int] = set()
    for section in ("history", "backtests", "predictions", "retrospectives"):
        directory = processed / section
        if directory.exists():
            seasons.update(
                int(path.name)
                for path in directory.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
    return sorted(seasons, reverse=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_backtest(processed: Path, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = processed / "backtests" / str(season)
    return _read_csv(base / "predictions.csv"), _read_csv(base / "round_metrics.csv")


def load_model_summary(processed: Path, season: int) -> pd.DataFrame:
    return _read_csv(processed / "backtests" / str(season) / "model_summary.csv")


def load_history_round(processed: Path, season: int, round_number: int) -> pd.DataFrame:
    return _read_csv(
        processed
        / "history"
        / str(season)
        / f"round_{round_number:02d}_training_rows.csv"
    )


def load_saved_prediction(processed: Path, season: int, round_number: int) -> pd.DataFrame:
    return _read_csv(
        processed
        / "predictions"
        / str(season)
        / f"round_{round_number:02d}_prediction.csv"
    )


def load_saved_retrospective(processed: Path, season: int, round_number: int) -> pd.DataFrame:
    return _read_csv(
        processed
        / "retrospectives"
        / str(season)
        / f"round_{round_number:02d}_retrospective.csv"
    )


def available_rounds(processed: Path, season: int, backtest: pd.DataFrame) -> list[int]:
    rounds: set[int] = set()
    if not backtest.empty and "RoundNumber" in backtest:
        rounds.update(pd.to_numeric(backtest["RoundNumber"], errors="coerce").dropna().astype(int))
    for section, suffix in (
        ("history", "_training_rows.csv"),
        ("predictions", "_prediction.csv"),
        ("retrospectives", "_retrospective.csv"),
    ):
        directory = processed / section / str(season)
        if directory.exists():
            for path in directory.glob(f"round_*{suffix}"):
                number = path.name.removeprefix("round_").removesuffix(suffix)
                if number.isdigit():
                    rounds.add(int(number))
    return sorted(rounds)


def build_race_view(
    *,
    saved_prediction: pd.DataFrame,
    backtest_prediction: pd.DataFrame,
    history: pd.DataFrame,
    retrospective_prediction: pd.DataFrame | None = None,
) -> RaceView:
    """Canlı kaydı önceler; sonucu yalnız ayrı geçmiş tablosundan eşler."""
    retrospective = (
        retrospective_prediction
        if retrospective_prediction is not None
        else pd.DataFrame()
    )
    if not saved_prediction.empty:
        prediction = saved_prediction.copy()
        source = "Kaydedilmiş yarış öncesi tahmin"
        rank_column = "PredictedPosition"
    elif not retrospective.empty:
        prediction = retrospective.copy()
        source = "Yarış sonrası geriye dönük deneme"
        rank_column = "PredictedPosition"
    elif not backtest_prediction.empty:
        prediction = backtest_prediction.copy()
        source = "Geriye dönük test tahmini"
        rank_column = "PenaltyAwareHybridPredictedRank"
    else:
        prediction = pd.DataFrame()
        source = None
        rank_column = ""

    event_frame = prediction if not prediction.empty else history
    event_name = (
        str(event_frame["EventName"].iloc[0])
        if not event_frame.empty and "EventName" in event_frame
        else "Yarış"
    )

    if prediction.empty:
        if history.empty:
            return RaceView(pd.DataFrame(), None, event_name, None, None)
        columns = [
            column
            for column in ("DriverId", "Driver", "Team", "StartOrder", "ActualPosition")
            if column in history
        ]
        rows = history[columns].copy()
        if "ActualPosition" in rows:
            rows = rows.sort_values("ActualPosition")
        return RaceView(rows.reset_index(drop=True), None, event_name, None, None)

    required = {"DriverId", "Driver", "Team", "StartOrder", rank_column}
    missing = sorted(required.difference(prediction.columns))
    if missing:
        raise ValueError(f"Tahmin dosyasında eksik sütunlar: {', '.join(missing)}")
    if prediction["DriverId"].duplicated().any():
        raise ValueError("Tahmin dosyasında pilot tekrarı var")

    rows = prediction[
        ["DriverId", "Driver", "Team", "StartOrder", rank_column]
    ].rename(columns={rank_column: "PredictedPosition"})
    rows["PredictedPosition"] = pd.to_numeric(
        rows["PredictedPosition"], errors="raise"
    ).astype(int)

    if not history.empty:
        result_columns = [
            column
            for column in ("DriverId", "ActualPosition", "CleanMAEEligible", "RaceStatusCategory")
            if column in history
        ]
        actual = history[result_columns].copy()
        if actual["DriverId"].duplicated().any():
            raise ValueError("Yarış sonucunda pilot tekrarı var")
        rows = rows.merge(actual, on="DriverId", how="left", validate="one_to_one")

    mae = None
    clean_mae = None
    if "ActualPosition" in rows:
        rows["ActualPosition"] = pd.to_numeric(rows["ActualPosition"], errors="coerce")
        rows["AbsoluteRankError"] = (
            rows["PredictedPosition"] - rows["ActualPosition"]
        ).abs()
        if rows["ActualPosition"].notna().any():
            mae = float(rows["AbsoluteRankError"].mean())
            if "CleanMAEEligible" in rows:
                clean = rows["CleanMAEEligible"].astype("boolean").fillna(False)
                if clean.any():
                    clean_mae = float(rows.loc[clean, "AbsoluteRankError"].mean())

    rows = rows.sort_values("PredictedPosition").reset_index(drop=True)
    return RaceView(rows, source, event_name, mae, clean_mae)
