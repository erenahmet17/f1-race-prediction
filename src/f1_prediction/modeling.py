"""Sızıntısız yarış sıralaması modelleri ve walk-forward backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .collection import WeekendDataPaths


class ModelingError(RuntimeError):
    """Model verisi veya bağımlılıkları güvenli çalışmaya uygun olmadığında oluşur."""


CATEGORICAL_FEATURES = (
    "DriverId",
    "Team",
    "TrackId",
    "QualiStageReached",
    "QualifyingDataQuality",
    "PracticeDataQuality",
    "SprintDataQuality",
    "SprintStatus",
    "TrackGeometryDataQuality",
    "GridPenaltyReason",
)

NUMERIC_FEATURES = (
    "QualifyingPositionPct",
    "QualiGapPct",
    "Q1GapPct",
    "Q2GapPct",
    "Q3GapPct",
    "QualiGapToTeammatePct",
    "QualifyingPositionToTeammatePct",
    "TeamQualifyingPositionMeanPct",
    "StartOrderPct",
    "StartsFromPitLane",
    "GridVsQualifyingDeltaPct",
    "GridPenaltyPositionsPct",
    "GridPenaltyNotePresent",
    "GridPenaltyApplied",
    "GridPenaltyPowerUnit",
    "GridPenaltySporting",
    "GridPenaltyParcFerme",
    "GridPenaltyTechnical",
    "GridPenaltyPitLane",
    "GridPermissionToStart",
    "GridPenaltyOther",
    "PracticePaceDeltaPct",
    "PracticeConsistencyPct",
    "PracticeLongRunLapCount",
    "PracticeStintCount",
    "PracticeSessionsUsed",
    "PracticeReferenceFallbackPct",
    "PracticeDataReliability",
    "PracticeDataMissing",
    "PracticeDriverDataAvailable",
    "TeamPracticePaceDeltaPct",
    "TeamPracticeDriverCount",
    "TeamPracticeLongRunLapCount",
    "TeamPracticeDataReliability",
    "PracticeTeamDataAvailable",
    "TeamPracticeDataMissing",
    "SprintWeekend",
    "SprintDataAvailable",
    "SprintDataMissing",
    "SprintStartPositionPct",
    "SprintFinishPositionPct",
    "SprintPositionsGainedPct",
    "SprintPaceDeltaPct",
    "SprintConsistencyPct",
    "SprintCleanLapCount",
    "SprintLapsCompletedPct",
    "SprintDataReliability",
    "SprintDNF",
    "PreRaceDataMissing",
    "TrackLapDistanceKm",
    "TrackLapDistanceApproximate",
    "TrackCornerCount",
    "TrackCornersPerKm",
    "TrackGeometryDataMissing",
    "TrackHistoricalPositionChangePct",
    "TrackHistoricalPositionChangeRelative",
    "TrackHistoricalGridRetention",
    "TrackHistoryRaceCount",
    "TrackHistoryDriverSampleCount",
    "TrackHistoryReliability",
    "TrackHistoryDataMissing",
    "DriverPriorRoundCount",
    "DriverPriorMeanFinishPct",
    "DriverPriorCleanMeanFinishPct",
    "DriverPriorDNFRate",
    "DriverPriorMeanStartPct",
    "DriverPriorMeanPositionsGainedPct",
    "DriverLast3MeanFinishPct",
    "TeamPriorRoundCount",
    "TeamPriorMeanFinishPct",
    "TeamPriorCleanMeanFinishPct",
    "TeamPriorDNFRate",
    "TeamPriorMeanStartPct",
    "TeamPriorMeanPositionsGainedPct",
    "TeamLast3MeanFinishPct",
)

MODEL_FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
GRID_DELTA_WEIGHT_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_GRID_DELTA_WEIGHT = 0.5
FORBIDDEN_MODEL_COLUMNS = {
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
}

_PRIOR_FEATURES = tuple(
    f"{prefix}{suffix}"
    for prefix in ("Driver", "Team")
    for suffix in (
        "PriorRoundCount",
        "PriorMeanFinishPct",
        "PriorCleanMeanFinishPct",
        "PriorDNFRate",
        "PriorMeanStartPct",
        "PriorMeanPositionsGainedPct",
        "Last3MeanFinishPct",
    )
)


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    round_metrics: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SavedBacktest:
    predictions_path: Path
    round_metrics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class UpcomingPredictionResult:
    predictions: pd.DataFrame
    grid_delta_weight: float
    training_rounds: int
    training_rows: int


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna(False).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )


def load_season_training_history(
    paths: WeekendDataPaths,
    *,
    season: int,
) -> pd.DataFrame:
    """Bir sezona ait doğrulanmış eğitim CSV'lerini round sırasıyla yükler."""
    directory = paths.processed / "history" / str(season)
    files = sorted(directory.glob("round_*_training_rows.csv"))
    if not files:
        raise ModelingError(f"{season} için eğitim geçmişi bulunamadı: {directory}")
    frames = [pd.read_csv(path) for path in files]
    history = pd.concat(frames, ignore_index=True)
    required = {
        "Season",
        "RoundNumber",
        "DriverId",
        "Driver",
        "Team",
        "StartOrder",
        "StartOrderPct",
        "ActualPosition",
        "ActualPositionPct",
        "CleanMAEEligible",
        "RaceDNF",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ModelingError(
            f"Eğitim geçmişinde eksik sütunlar var: {', '.join(missing)}"
        )
    history["Season"] = pd.to_numeric(history["Season"], errors="coerce")
    history["RoundNumber"] = pd.to_numeric(
        history["RoundNumber"], errors="coerce"
    )
    if history[["Season", "RoundNumber"]].isna().any().any():
        raise ModelingError("Eğitim geçmişinde geçersiz sezon/round var")
    if set(history["Season"].astype(int)) != {int(season)}:
        raise ModelingError("Eğitim geçmişinde başka sezona ait satırlar var")
    duplicates = history.duplicated(["Season", "RoundNumber", "DriverId"])
    if duplicates.any():
        raise ModelingError("Eğitim geçmişinde aynı round için pilot tekrarı var")
    return history.sort_values(
        ["Season", "RoundNumber", "StartOrder"]
    ).reset_index(drop=True)


def _prior_summary(
    past: pd.DataFrame,
    *,
    key: str,
    prefix: str,
) -> pd.DataFrame:
    work = past.copy()
    work["_Completed"] = _as_bool(work["CleanMAEEligible"])
    work["_DNF"] = _as_bool(work["RaceDNF"])
    work["_PositionsGainedPct"] = (
        pd.to_numeric(work["StartOrderPct"], errors="coerce")
        - pd.to_numeric(work["ActualPositionPct"], errors="coerce")
    )
    grouped = work.groupby(key, dropna=False)
    summary = grouped.agg(
        **{
            f"{prefix}PriorRoundCount": ("RoundNumber", "nunique"),
            f"{prefix}PriorMeanFinishPct": ("ActualPositionPct", "mean"),
            f"{prefix}PriorDNFRate": ("_DNF", "mean"),
            f"{prefix}PriorMeanStartPct": ("StartOrderPct", "mean"),
            f"{prefix}PriorMeanPositionsGainedPct": (
                "_PositionsGainedPct",
                "mean",
            ),
        }
    )
    clean = work[work["_Completed"]]
    clean_mean = clean.groupby(key, dropna=False)["ActualPositionPct"].mean()
    summary[f"{prefix}PriorCleanMeanFinishPct"] = clean_mean

    per_round = (
        work.groupby([key, "RoundNumber"], dropna=False)["ActualPositionPct"]
        .mean()
        .rename("_RoundMeanFinishPct")
        .reset_index()
        .sort_values([key, "RoundNumber"])
    )
    last_three = (
        per_round.groupby(key, group_keys=False)
        .tail(3)
        .groupby(key, dropna=False)["_RoundMeanFinishPct"]
        .mean()
    )
    summary[f"{prefix}Last3MeanFinishPct"] = last_three
    return summary


def add_prior_form_features(history: pd.DataFrame) -> pd.DataFrame:
    """Her rounda yalnız daha önce tamamlanmış roundlardan form özellikleri ekler."""
    required = {
        "Season",
        "RoundNumber",
        "DriverId",
        "Team",
        "StartOrderPct",
        "ActualPositionPct",
        "CleanMAEEligible",
        "RaceDNF",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ModelingError(f"Form hesabında eksik sütunlar var: {', '.join(missing)}")

    ordered = history.copy().reset_index(drop=False).rename(
        columns={"index": "_OriginalOrder"}
    )
    output: list[pd.DataFrame] = []
    for _, season_rows in ordered.groupby("Season", sort=True):
        past_frames: list[pd.DataFrame] = []
        for _, current in season_rows.groupby("RoundNumber", sort=True):
            current = current.copy()
            for column in _PRIOR_FEATURES:
                current[column] = np.nan
            current["DriverPriorRoundCount"] = 0
            current["TeamPriorRoundCount"] = 0
            if past_frames:
                past = pd.concat(past_frames, ignore_index=True)
                for key, prefix in (("DriverId", "Driver"), ("Team", "Team")):
                    summary = _prior_summary(past, key=key, prefix=prefix)
                    current = current.merge(
                        summary,
                        left_on=key,
                        right_index=True,
                        how="left",
                        suffixes=("", "_new"),
                        validate="many_to_one",
                    )
                    for column in summary.columns:
                        new_column = f"{column}_new"
                        current[column] = current[new_column]
                        current = current.drop(columns=new_column)
                    count_column = f"{prefix}PriorRoundCount"
                    current[count_column] = current[count_column].fillna(0).astype(int)
            output.append(current)
            past_frames.append(current)
    return (
        pd.concat(output, ignore_index=True)
        .sort_values("_OriginalOrder")
        .drop(columns="_OriginalOrder")
        .reset_index(drop=True)
    )


def build_model_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Yalnız yarış öncesinde bilinen, açık izin listeli model matrisini üretir."""
    work = rows.copy()
    race_size = work.groupby(["Season", "RoundNumber"])["DriverId"].transform(
        "count"
    )
    denominator = (race_size - 1).replace(0, np.nan)
    work["QualifyingPositionToTeammatePct"] = pd.to_numeric(
        work["QualifyingPositionToTeammate"], errors="coerce"
    ) / denominator
    work["TeamQualifyingPositionMeanPct"] = (
        pd.to_numeric(work["TeamQualifyingPositionMean"], errors="coerce") - 1
    ) / denominator
    work["SprintStartPositionPct"] = (
        pd.to_numeric(work["SprintStartPosition"], errors="coerce") - 1
    ) / denominator
    work["SprintFinishPositionPct"] = (
        pd.to_numeric(work["SprintFinishPosition"], errors="coerce") - 1
    ) / denominator
    work["SprintPositionsGainedPct"] = pd.to_numeric(
        work["SprintPositionsGained"], errors="coerce"
    ) / denominator
    work["GridPenaltyPositionsPct"] = pd.to_numeric(
        work["GridPenaltyPositions"], errors="coerce"
    ) / denominator

    missing = sorted(set(MODEL_FEATURES).difference(work.columns))
    if missing:
        raise ModelingError(f"Model özelliklerinde eksik sütunlar var: {', '.join(missing)}")
    if FORBIDDEN_MODEL_COLUMNS.intersection(MODEL_FEATURES):
        raise AssertionError("Yarış sonrası sütunu model izin listesine girdi")

    features = work.loc[:, MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        values = features[column].astype("string").str.strip()
        features[column] = values.mask(values.isna() | values.eq(""), "__MISSING__")
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def rank_scores(
    scores: np.ndarray | pd.Series,
    *,
    start_order: pd.Series,
    driver_ids: pd.Series,
) -> pd.Series:
    """Büyük skor daha iyi olacak biçimde deterministik 1..N sırası üretir."""
    ranking = pd.DataFrame(
        {
            "Score": np.asarray(scores, dtype=float),
            "StartOrder": pd.to_numeric(start_order, errors="coerce"),
            "DriverId": driver_ids.astype(str),
        },
        index=start_order.index,
    )
    if ranking["Score"].isna().any():
        raise ModelingError("Model NaN skor üretti")
    ordered = ranking.sort_values(
        ["Score", "StartOrder", "DriverId"],
        ascending=[False, True, True],
    )
    ranks = pd.Series(index=ranking.index, dtype=int)
    ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks.astype(int)


def grid_delta_scores(
    predicted_positions_gained_pct: np.ndarray | pd.Series,
    *,
    start_order_pct: pd.Series,
    weight: float = 1.0,
) -> np.ndarray:
    """Beklenen sıra değişimini nihai sıralama skoruna dönüştürür.

    Skor, tahmini bitiş yüzdesinin negatifidir. Değerler özellikle kırpılmaz;
    böylece ceza nedeniyle geriden başlayan güçlü bir pilotun büyük bir ilerleme
    yapmasına model izin verebilir.
    """
    if weight < 0:
        raise ValueError("Grid-delta ağırlığı negatif olamaz")
    predicted_delta = np.asarray(predicted_positions_gained_pct, dtype=float)
    start = pd.to_numeric(start_order_pct, errors="coerce").to_numpy(dtype=float)
    if np.isnan(predicted_delta).any() or np.isnan(start).any():
        raise ModelingError("Grid-delta modeli NaN skor üretti")
    return (weight * predicted_delta) - start


def select_grid_delta_weight(previous_predictions: pd.DataFrame) -> float:
    """Yalnız geçmiş out-of-sample tahminlerle grid-delta ağırlığını seçer."""
    if previous_predictions.empty:
        return DEFAULT_GRID_DELTA_WEIGHT
    required = {
        "RoundNumber",
        "DriverId",
        "StartOrder",
        "ActualPosition",
        "ExpectedPositionsGainedPct",
    }
    missing = sorted(required.difference(previous_predictions.columns))
    if missing:
        raise ModelingError(
            f"Grid-delta kalibrasyonunda eksik sütunlar var: {', '.join(missing)}"
        )

    candidate_errors: list[tuple[float, float]] = []
    for weight in GRID_DELTA_WEIGHT_CANDIDATES:
        absolute_errors: list[float] = []
        for _, round_rows in previous_predictions.groupby("RoundNumber", sort=True):
            denominator = max(len(round_rows) - 1, 1)
            start_pct = (
                pd.to_numeric(round_rows["StartOrder"], errors="raise") - 1.0
            ) / denominator
            scores = grid_delta_scores(
                round_rows["ExpectedPositionsGainedPct"],
                start_order_pct=start_pct,
                weight=weight,
            )
            ranks = rank_scores(
                scores,
                start_order=round_rows["StartOrder"],
                driver_ids=round_rows["DriverId"],
            )
            errors = (
                ranks
                - pd.to_numeric(round_rows["ActualPosition"], errors="raise")
            ).abs()
            absolute_errors.extend(errors.astype(float).tolist())
        candidate_errors.append((float(np.mean(absolute_errors)), weight))

    _, selected = min(
        candidate_errors,
        key=lambda item: (
            item[0],
            abs(item[1] - DEFAULT_GRID_DELTA_WEIGHT),
            item[1],
        ),
    )
    return selected


def penalty_aware_hybrid_scores(
    *,
    catboost_ranks: pd.Series,
    grid_delta_ranks: pd.Series,
    penalty_applied: pd.Series,
) -> np.ndarray:
    """Cezalı pilotlarda doğrudan performans, diğerlerinde grid-delta sırasını kullanır."""
    direct = pd.to_numeric(catboost_ranks, errors="coerce").to_numpy(dtype=float)
    grid_delta = pd.to_numeric(grid_delta_ranks, errors="coerce").to_numpy(
        dtype=float
    )
    penalized = _as_bool(penalty_applied).to_numpy(dtype=bool)
    if np.isnan(direct).any() or np.isnan(grid_delta).any():
        raise ModelingError("Ceza duyarlı hibrit model NaN sıra aldı")
    return -np.where(penalized, direct, grid_delta)


def _ridge_scores(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
) -> np.ndarray:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ModelingError("Ridge baseline için scikit-learn kurulmalı") from exc

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
    model = Pipeline([("features", transform), ("model", Ridge(alpha=10.0))])
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=float)


def _catboost_scores(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_groups: pd.Series,
    test_x: pd.DataFrame,
) -> np.ndarray:
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise ModelingError(
            "CatBoostRanker için 'pip install catboost' çalıştırılmalı"
        ) from exc

    pool = Pool(
        train_x,
        label=train_y,
        group_id=train_groups.astype(str),
        cat_features=list(CATEGORICAL_FEATURES),
    )
    model = CatBoostRanker(
        loss_function="QueryRMSE",
        iterations=350,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(pool)
    return np.asarray(model.predict(test_x), dtype=float)


def _catboost_grid_delta_scores(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
) -> np.ndarray:
    """Beklenen sıra kazanımını ve bundan türetilen bitiş skorunu üretir."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ModelingError(
            "CatBoost grid-delta modeli için 'pip install catboost' çalıştırılmalı"
        ) from exc

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=350,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        cat_features=list(CATEGORICAL_FEATURES),
    )
    model.fit(train_x, train_y)
    predicted_delta = np.asarray(model.predict(test_x), dtype=float)
    return predicted_delta


def _round_metric_rows(
    predictions: pd.DataFrame,
    *,
    training_rounds: int,
    training_rows: int,
) -> list[dict[str, object]]:
    clean = _as_bool(predictions["CleanMAEEligible"])
    rows: list[dict[str, object]] = []
    for model_name, rank_column in (
        ("Grid", "GridPredictedRank"),
        ("Ridge", "RidgePredictedRank"),
        ("CatBoostRanker", "CatBoostPredictedRank"),
        ("CatBoostGridDelta", "CatBoostGridDeltaPredictedRank"),
        ("PenaltyAwareHybrid", "PenaltyAwareHybridPredictedRank"),
    ):
        error = (
            pd.to_numeric(predictions[rank_column], errors="coerce")
            - pd.to_numeric(predictions["ActualPosition"], errors="coerce")
        ).abs()
        rows.append(
            {
                "Season": int(predictions["Season"].iloc[0]),
                "RoundNumber": int(predictions["RoundNumber"].iloc[0]),
                "EventName": str(predictions["EventName"].iloc[0]),
                "Model": model_name,
                "MAE": float(error.mean()),
                "CleanMAE": float(error[clean].mean()),
                "DriverCount": int(len(predictions)),
                "CleanDriverCount": int(clean.sum()),
                "TrainingRounds": int(training_rounds),
                "TrainingRows": int(training_rows),
            }
        )
    return rows


def walk_forward_backtest(
    history: pd.DataFrame,
    *,
    min_train_rounds: int = 4,
) -> BacktestResult:
    """Her roundu yalnız önceki roundlarla eğiterek modelleri karşılaştırır."""
    if min_train_rounds < 1:
        raise ValueError("min_train_rounds en az 1 olmalı")
    enriched = add_prior_form_features(history)
    rounds = sorted(pd.to_numeric(enriched["RoundNumber"], errors="raise").unique())
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for target_round in rounds:
        prior_rounds = [round_number for round_number in rounds if round_number < target_round]
        if len(prior_rounds) < min_train_rounds:
            continue
        train = enriched[enriched["RoundNumber"].isin(prior_rounds)].copy()
        test = enriched[enriched["RoundNumber"] == target_round].copy()
        train = train.sort_values(["Season", "RoundNumber", "StartOrder"])
        test = test.sort_values("StartOrder")
        train_x = build_model_features(train)
        test_x = build_model_features(test)
        train_y = 1.0 - pd.to_numeric(train["ActualPositionPct"], errors="raise")
        train_groups = (
            train["Season"].astype(int).astype(str)
            + "_"
            + train["RoundNumber"].astype(int).astype(str).str.zfill(2)
        )

        ridge_scores = _ridge_scores(train_x, train_y, test_x)
        catboost_scores = _catboost_scores(
            train_x,
            train_y,
            train_groups,
            test_x,
        )
        train_grid_delta_y = (
            pd.to_numeric(train["StartOrderPct"], errors="raise")
            - pd.to_numeric(train["ActualPositionPct"], errors="raise")
        )
        predicted_grid_delta_pct = _catboost_grid_delta_scores(
            train_x,
            train_grid_delta_y,
            test_x,
        )
        previous_predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        grid_delta_weight = select_grid_delta_weight(previous_predictions)
        catboost_grid_delta_scores = grid_delta_scores(
            predicted_grid_delta_pct,
            start_order_pct=test["StartOrderPct"],
            weight=grid_delta_weight,
        )
        result = test[
            [
                "Season",
                "RoundNumber",
                "EventName",
                "DriverId",
                "Driver",
                "Team",
                "QualifyingPosition",
                "StartOrder",
                "GridVsQualifyingDelta",
                "GridPenaltyApplied",
                "GridPenaltyPositions",
                "GridPenaltyReason",
                "ActualPosition",
                "CleanMAEEligible",
            ]
        ].copy()
        result["GridScore"] = -pd.to_numeric(test["StartOrder"], errors="raise")
        result["RidgeScore"] = ridge_scores
        result["CatBoostScore"] = catboost_scores
        result["ExpectedPositionsGainedPct"] = predicted_grid_delta_pct
        result["ExpectedPositionsGained"] = predicted_grid_delta_pct * (
            len(test) - 1
        )
        result["GridDeltaWeight"] = grid_delta_weight
        result["WeightedExpectedPositionsGained"] = (
            result["ExpectedPositionsGained"] * grid_delta_weight
        )
        result["CatBoostGridDeltaScore"] = catboost_grid_delta_scores
        result["GridPredictedRank"] = pd.to_numeric(
            test["StartOrder"], errors="raise"
        ).astype(int)
        result["RidgePredictedRank"] = rank_scores(
            ridge_scores,
            start_order=test["StartOrder"],
            driver_ids=test["DriverId"],
        )
        result["CatBoostPredictedRank"] = rank_scores(
            catboost_scores,
            start_order=test["StartOrder"],
            driver_ids=test["DriverId"],
        )
        result["CatBoostGridDeltaPredictedRank"] = rank_scores(
            catboost_grid_delta_scores,
            start_order=test["StartOrder"],
            driver_ids=test["DriverId"],
        )
        penalty_hybrid_scores = penalty_aware_hybrid_scores(
            catboost_ranks=result["CatBoostPredictedRank"],
            grid_delta_ranks=result["CatBoostGridDeltaPredictedRank"],
            penalty_applied=result["GridPenaltyApplied"],
        )
        result["PenaltyAwareHybridScore"] = penalty_hybrid_scores
        result["PenaltyAwareHybridPredictedRank"] = rank_scores(
            penalty_hybrid_scores,
            start_order=test["StartOrder"],
            driver_ids=test["DriverId"],
        )
        metric_rows.extend(
            _round_metric_rows(
                result,
                training_rounds=len(prior_rounds),
                training_rows=len(train),
            )
        )
        prediction_frames.append(result)

    if not prediction_frames:
        raise ModelingError("Backtest için yeterli geçmiş round yok")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    round_metrics = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, object]] = []
    for model_name, rank_column in (
        ("Grid", "GridPredictedRank"),
        ("Ridge", "RidgePredictedRank"),
        ("CatBoostRanker", "CatBoostPredictedRank"),
        ("CatBoostGridDelta", "CatBoostGridDeltaPredictedRank"),
        ("PenaltyAwareHybrid", "PenaltyAwareHybridPredictedRank"),
    ):
        error = (
            predictions[rank_column] - predictions["ActualPosition"]
        ).abs()
        clean = _as_bool(predictions["CleanMAEEligible"])
        model_rounds = round_metrics[round_metrics["Model"] == model_name]
        summary_rows.append(
            {
                "Model": model_name,
                "OverallMAE": float(error.mean()),
                "OverallCleanMAE": float(error[clean].mean()),
                "MeanRoundMAE": float(model_rounds["MAE"].mean()),
                "MeanRoundCleanMAE": float(model_rounds["CleanMAE"].mean()),
                "RoundsEvaluated": int(model_rounds["RoundNumber"].nunique()),
                "DriversEvaluated": int(len(predictions)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["OverallMAE", "OverallCleanMAE"]
    ).reset_index(drop=True)
    return BacktestResult(
        predictions=predictions,
        round_metrics=round_metrics,
        summary=summary,
    )


def predict_upcoming_round(
    history: pd.DataFrame,
    upcoming: pd.DataFrame,
    *,
    calibration_predictions: pd.DataFrame | None = None,
    min_train_rounds: int = 4,
) -> UpcomingPredictionResult:
    """Yalnız tamamlanmış geçmiş yarışlarla bir sonraki roundu tahmin eder."""
    if min_train_rounds < 1:
        raise ValueError("min_train_rounds en az 1 olmalı")
    required_upcoming = {
        "Season",
        "RoundNumber",
        "EventName",
        "DriverId",
        "Driver",
        "Team",
        "QualifyingPosition",
        "StartOrder",
        "StartOrderPct",
        "GridVsQualifyingDelta",
        "GridPenaltyApplied",
        "GridPenaltyPositions",
        "GridPenaltyReason",
    }
    missing = sorted(required_upcoming.difference(upcoming.columns))
    if missing:
        raise ModelingError(
            f"Tahmin hafta sonu tablosunda eksik sütunlar var: {', '.join(missing)}"
        )
    if upcoming.empty:
        raise ModelingError("Tahmin hafta sonu tablosu boş")
    if upcoming["DriverId"].duplicated().any():
        raise ModelingError("Tahmin hafta sonu tablosunda pilot tekrarı var")
    seasons = set(pd.to_numeric(upcoming["Season"], errors="coerce").dropna())
    rounds = set(pd.to_numeric(upcoming["RoundNumber"], errors="coerce").dropna())
    if len(seasons) != 1 or len(rounds) != 1:
        raise ModelingError("Tahmin tablosu tam olarak bir sezon ve round içermeli")
    target_season = int(next(iter(seasons)))
    target_round = int(next(iter(rounds)))

    for column in FORBIDDEN_MODEL_COLUMNS:
        if column in upcoming and upcoming[column].notna().any():
            raise ModelingError(
                f"Yarış öncesi tahmin tablosunda yarış sonrası sütunu dolu: {column}"
            )

    required_history = {
        "Season",
        "RoundNumber",
        "DriverId",
        "Team",
        "StartOrder",
        "StartOrderPct",
        "ActualPosition",
        "ActualPositionPct",
        "CleanMAEEligible",
        "RaceDNF",
    }
    missing_history = sorted(required_history.difference(history.columns))
    if missing_history:
        raise ModelingError(
            f"Tahmin geçmişinde eksik sütunlar var: {', '.join(missing_history)}"
        )
    history_seasons = set(
        pd.to_numeric(history["Season"], errors="coerce").dropna().astype(int)
    )
    if history_seasons != {target_season}:
        raise ModelingError("Tahmin geçmişi hedef sezonla uyuşmuyor")
    history_rounds = sorted(
        pd.to_numeric(history["RoundNumber"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(history_rounds) < min_train_rounds:
        raise ModelingError(
            f"Tahmin için en az {min_train_rounds} tamamlanmış round gerekli"
        )
    if not history_rounds or max(history_rounds) >= target_round:
        raise ModelingError("Tahmin geçmişi hedef veya gelecek round içeriyor")

    target = upcoming.copy()
    target["ActualPosition"] = np.nan
    target["ActualPositionPct"] = np.nan
    target["CleanMAEEligible"] = False
    target["RaceDNF"] = False
    combined = pd.concat([history.copy(), target], ignore_index=True, sort=False)
    enriched = add_prior_form_features(combined)
    train = enriched[enriched["RoundNumber"] < target_round].copy()
    test = enriched[enriched["RoundNumber"] == target_round].copy()
    train = train.sort_values(["Season", "RoundNumber", "StartOrder"])
    test = test.sort_values("StartOrder")

    train_x = build_model_features(train)
    test_x = build_model_features(test)
    train_y = 1.0 - pd.to_numeric(train["ActualPositionPct"], errors="raise")
    train_groups = (
        train["Season"].astype(int).astype(str)
        + "_"
        + train["RoundNumber"].astype(int).astype(str).str.zfill(2)
    )
    ridge_scores = _ridge_scores(train_x, train_y, test_x)
    catboost_scores = _catboost_scores(train_x, train_y, train_groups, test_x)
    train_grid_delta_y = (
        pd.to_numeric(train["StartOrderPct"], errors="raise")
        - pd.to_numeric(train["ActualPositionPct"], errors="raise")
    )
    predicted_grid_delta_pct = _catboost_grid_delta_scores(
        train_x,
        train_grid_delta_y,
        test_x,
    )

    calibration = (
        calibration_predictions.copy()
        if calibration_predictions is not None
        else pd.DataFrame()
    )
    if not calibration.empty:
        calibration_rounds = pd.to_numeric(
            calibration["RoundNumber"], errors="raise"
        )
        if (calibration_rounds >= target_round).any():
            raise ModelingError("Kalibrasyon tahminleri hedef veya gelecek round içeriyor")
    grid_delta_weight = select_grid_delta_weight(calibration)
    catboost_grid_delta_scores = grid_delta_scores(
        predicted_grid_delta_pct,
        start_order_pct=test["StartOrderPct"],
        weight=grid_delta_weight,
    )

    output_columns = [
        "Season",
        "RoundNumber",
        "EventName",
        "DriverId",
        "Driver",
        "Team",
        "QualifyingPosition",
        "StartOrder",
        "GridVsQualifyingDelta",
        "GridPenaltyApplied",
        "GridPenaltyPositions",
        "GridPenaltyReason",
        "GridStatus",
        "GridPublishedAt",
        "GridDocumentNumber",
        "QualifyingDataQuality",
        "PracticeDataQuality",
        "SprintDataQuality",
        "PreRaceDataMissing",
    ]
    available_output_columns = [
        column for column in output_columns if column in test.columns
    ]
    result = test[available_output_columns].copy()
    result["GridScore"] = -pd.to_numeric(test["StartOrder"], errors="raise")
    result["RidgeScore"] = ridge_scores
    result["CatBoostScore"] = catboost_scores
    result["ExpectedPositionsGainedPct"] = predicted_grid_delta_pct
    result["ExpectedPositionsGained"] = predicted_grid_delta_pct * (len(test) - 1)
    result["GridDeltaWeight"] = grid_delta_weight
    result["WeightedExpectedPositionsGained"] = (
        result["ExpectedPositionsGained"] * grid_delta_weight
    )
    result["CatBoostGridDeltaScore"] = catboost_grid_delta_scores
    result["GridPredictedRank"] = pd.to_numeric(
        test["StartOrder"], errors="raise"
    ).astype(int)
    result["RidgePredictedRank"] = rank_scores(
        ridge_scores,
        start_order=test["StartOrder"],
        driver_ids=test["DriverId"],
    )
    result["CatBoostPredictedRank"] = rank_scores(
        catboost_scores,
        start_order=test["StartOrder"],
        driver_ids=test["DriverId"],
    )
    result["CatBoostGridDeltaPredictedRank"] = rank_scores(
        catboost_grid_delta_scores,
        start_order=test["StartOrder"],
        driver_ids=test["DriverId"],
    )
    penalty_hybrid_scores = penalty_aware_hybrid_scores(
        catboost_ranks=result["CatBoostPredictedRank"],
        grid_delta_ranks=result["CatBoostGridDeltaPredictedRank"],
        penalty_applied=result["GridPenaltyApplied"],
    )
    result["PenaltyAwareHybridScore"] = penalty_hybrid_scores
    result["PenaltyAwareHybridPredictedRank"] = rank_scores(
        penalty_hybrid_scores,
        start_order=test["StartOrder"],
        driver_ids=test["DriverId"],
    )
    result["PredictionModel"] = "PenaltyAwareHybrid"
    result["PredictedPosition"] = result["PenaltyAwareHybridPredictedRank"]
    result["TrainingThroughRound"] = max(history_rounds)
    result["TrainingRounds"] = len(history_rounds)
    result["TrainingRows"] = len(train)
    result["CreatedAtUtc"] = pd.Timestamp.now(tz="UTC").isoformat()
    result = result.sort_values("PredictedPosition").reset_index(drop=True)
    return UpcomingPredictionResult(
        predictions=result,
        grid_delta_weight=grid_delta_weight,
        training_rounds=len(history_rounds),
        training_rows=len(train),
    )


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_backtest(
    result: BacktestResult,
    *,
    paths: WeekendDataPaths,
    season: int,
) -> SavedBacktest:
    directory = paths.processed / "backtests" / str(season)
    return SavedBacktest(
        predictions_path=_atomic_csv(
            result.predictions, directory / "predictions.csv"
        ),
        round_metrics_path=_atomic_csv(
            result.round_metrics, directory / "round_metrics.csv"
        ),
        summary_path=_atomic_csv(result.summary, directory / "model_summary.csv"),
    )
