from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import WeekendDataPaths  # noqa: E402
from f1_prediction.modeling import (  # noqa: E402
    load_season_training_history,
    save_backtest,
    walk_forward_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid, Ridge, CatBoostRanker, CatBoostGridDelta ve PenaltyAwareHybrid "
            "walk-forward karşılaştırması."
        )
    )
    parser.add_argument("season", type=int, help="Sezon yılı")
    parser.add_argument(
        "--min-train-rounds",
        type=int,
        default=4,
        help="İlk testten önce gereken tamamlanmış round sayısı (varsayılan: 4)",
    )
    args = parser.parse_args()

    paths = WeekendDataPaths.from_project_root(PROJECT_ROOT)
    history = load_season_training_history(paths, season=args.season)
    result = walk_forward_backtest(
        history,
        min_train_rounds=args.min_train_rounds,
    )
    saved = save_backtest(result, paths=paths, season=args.season)
    print(result.summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Özet: {saved.summary_path}")
    print(f"Round metrikleri: {saved.round_metrics_path}")
    print(f"Pilot tahminleri: {saved.predictions_path}")


if __name__ == "__main__":
    main()
