from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.backfill import backfill_training_history  # noqa: E402
from f1_prediction.collection import WeekendDataPaths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Güncel sezonun tamamlanan roundlarını eğitim geçmişine ekler."
    )
    parser.add_argument("season", type=int, help="Sezon yılı")
    parser.add_argument("through_round", type=int, help="Tamamlanmış son round")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Doğrulanmış mevcut dosyaları yeniden oluştur",
    )
    args = parser.parse_args()

    summary = backfill_training_history(
        args.season,
        args.through_round,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
        force=args.force,
    )
    print(f"Toplanan roundlar: {list(summary.collected_rounds)}")
    print(f"Doğrulanıp atlanan roundlar: {list(summary.skipped_rounds)}")


if __name__ == "__main__":
    main()
