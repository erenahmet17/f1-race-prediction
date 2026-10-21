from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_prediction.backfill import backfill_track_history  # noqa: E402
from f1_prediction.collection import WeekendDataPaths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eski yarışları yalnız pist karakteri için toplar."
    )
    parser.add_argument("start_season", type=int, help="İlk sezon")
    parser.add_argument("end_season", type=int, nargs="?", help="Son sezon")
    parser.add_argument(
        "--through-round",
        type=int,
        help="Yalnız tek sezonda bu rounda kadar işle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Doğrulanmış mevcut dosyaları yeniden oluştur",
    )
    args = parser.parse_args()
    end_season = args.end_season or args.start_season

    summaries = backfill_track_history(
        args.start_season,
        end_season,
        paths=WeekendDataPaths.from_project_root(PROJECT_ROOT),
        through_round=args.through_round,
        force=args.force,
    )
    for season, summary in summaries.items():
        print(
            f"{season}: {len(summary.collected_rounds)} toplandı, "
            f"{len(summary.skipped_rounds)} doğrulanıp atlandı"
        )


if __name__ == "__main__":
    main()
