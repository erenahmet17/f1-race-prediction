from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import (  # noqa: E402
    WeekendDataPaths,
    collect_pre_race_weekend,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir F1 yarış haftasının yarış öncesi özelliklerini toplar."
    )
    parser.add_argument("season", type=int, help="Sezon yılı, örneğin 2026")
    parser.add_argument("round_number", type=int, help="Round numarası")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Birleşmiş CSV dosyasını kaydetme",
    )
    args = parser.parse_args()

    result = collect_pre_race_weekend(
        args.season,
        args.round_number,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
        save=not args.no_save,
    )
    weekend_type = "sprint" if result.sessions.is_sprint_weekend else "normal"
    print(f"{result.event_name}: {weekend_type} hafta sonu")
    print(f"Pilot sayısı: {len(result.features)}")
    print(f"FIA final grid belgesi: {result.final_grid.document.document_number}")
    if result.feature_path is not None:
        print(f"Özellik tablosu: {result.feature_path}")


if __name__ == "__main__":
    main()
