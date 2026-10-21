from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.collection import WeekendDataPaths  # noqa: E402
from f1_prediction.history import collect_completed_round  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tamamlanan yarışı eğitim geçmişine ekler."
    )
    parser.add_argument("season", type=int, help="Sezon yılı, örneğin 2026")
    parser.add_argument("round_number", type=int, help="Tamamlanan round numarası")
    parser.add_argument("--no-save", action="store_true", help="CSV dosyalarını kaydetme")
    args = parser.parse_args()

    result = collect_completed_round(
        args.season,
        args.round_number,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
        save=not args.no_save,
    )
    clean_count = int(result.rows["CleanMAEEligible"].sum())
    print(f"{result.pre_race.event_name}: {len(result.rows)} yarış sonucu işlendi")
    print(f"FIA final yarış belgesi: {result.final_race.document.document_number}")
    print(f"Clean MAE için uygun pilot: {clean_count}")
    if result.history_path is not None:
        print(f"Geçmiş dosyası: {result.history_path}")


if __name__ == "__main__":
    main()
