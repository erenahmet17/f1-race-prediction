from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import WeekendDataPaths  # noqa: E402
from f1_prediction.prediction import create_pre_race_prediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final grid açıklandıktan sonra yarış öncesi veriyi toplar ve "
            "PenaltyAwareHybrid tahmini üretir."
        )
    )
    parser.add_argument("season", type=int, help="Sezon yılı, örneğin 2026")
    parser.add_argument("round_number", type=int, help="Tahmin edilecek round")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Hafta sonu özelliklerini ve tahmin CSV'sini kaydetme",
    )
    args = parser.parse_args()

    result = create_pre_race_prediction(
        args.season,
        args.round_number,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
        save=not args.no_save,
    )
    predictions = result.model.predictions
    print(f"{result.weekend.event_name} yarış tahmini")
    print(
        f"Eğitim: round 1-{predictions['TrainingThroughRound'].iloc[0]} "
        f"({result.model.training_rows} pilot satırı)"
    )
    print(f"Grid-delta ağırlığı: {result.model.grid_delta_weight:.2f}")
    practice_quality = result.weekend.features["PracticeDataQuality"].value_counts()
    print(f"Antrenman verisi kalitesi: {practice_quality.to_dict()}")
    print(
        predictions[
            [
                "PredictedPosition",
                "Driver",
                "Team",
                "StartOrder",
                "QualifyingPosition",
                "GridPenaltyReason",
            ]
        ].to_string(index=False)
    )
    if result.prediction_path is not None:
        print(f"Tahmin dosyası: {result.prediction_path}")


if __name__ == "__main__":
    main()
