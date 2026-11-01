from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import (  # noqa: E402
    WeekendDataPaths,
    cache_weekend_laps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Antrenman ve sprint turlarını önbelleğe indirir. OpenF1 canlı seans "
            "sırasında kapalı olduğu için sıralamadan sonra çalıştırın."
        )
    )
    parser.add_argument("season", type=int, help="Sezon yılı, örneğin 2026")
    parser.add_argument("round_number", type=int, help="Round numarası")
    args = parser.parse_args()

    lap_counts = cache_weekend_laps(
        args.season,
        args.round_number,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
    )
    for session_name, count in lap_counts.items():
        print(f"{session_name}: {count} tur")
    if not all(lap_counts.values()):
        print(
            "Bazı seansların turları alınamadı. Seans henüz bitmemiş veya OpenF1 "
            "canlı seans nedeniyle kapalı olabilir; sonra yeniden çalıştırın."
        )


if __name__ == "__main__":
    main()
