"""Full-network AEMET daily scrape — multi-period, resumable.

Daily is the big one. AEMET's daily climatological endpoint caps
each request at 180 days, so a single station's decade takes ~20
chunks — roughly 6× the monthly budget. Expect ~36 hours of wall
time at 120 req/min pacing for ~947 stations × 1920-2024.

The script walks the history in two-year periods so progress is
checkpointed every few hours. The archive is idempotent — interrupt
and re-run safely.

Because most stations don't have data before ~1950, running from
1950 onwards is usually the right call if quota is a concern (see
``--start``).

Run:
    uv run python scripts/aemet_daily.py                  # full history
    uv run python scripts/aemet_daily.py --start 1980     # 1980 onward
    uv run python scripts/aemet_daily.py --start 2015 --end 2024   # decade
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aemet_common import build_archive, setup_logging


# Two-year periods. Shorter than monthly because daily generates much
# more data per station-period — checkpointing to disk more often
# keeps the "progress lost on interrupt" bounded.
PERIODS: list[tuple[int, int]] = [
    (start, min(start + 1, 2024)) for start in range(1920, 2025, 2)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1920, help="first year to scrape")
    p.add_argument("--end", type=int, default=2024, help="last year to scrape")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("aemet_daily")
    archive = build_archive("daily")

    logger.info("refreshing station inventory")
    inventory = archive.sync_stations()
    logger.info(f"inventory: {len(inventory)} stations")

    t0 = time.monotonic()
    for i, (y1, y2) in enumerate(PERIODS, 1):
        if y2 < args.start or y1 > args.end:
            continue
        y1 = max(y1, args.start)
        y2 = min(y2, args.end)
        logger.info(f"period {i}/{len(PERIODS)}: {y1}-{y2}")
        period_t0 = time.monotonic()
        try:
            ds = archive.sync(
                "aemet_daily",
                stations=inventory,
                since=f"{y1}-01-01",
                until=f"{y2}-12-31",
            )
        except KeyboardInterrupt:
            logger.warning(f"interrupted in period {y1}-{y2}; archive is safe")
            raise
        logger.info(
            f"  period {y1}-{y2}: stations={ds.sizes['station']}, "
            f"days={ds.sizes['time']}, "
            f"elapsed={time.monotonic() - period_t0:.1f}s"
        )

    logger.info(f"all periods done in {time.monotonic() - t0:.1f}s")
    logger.info("archive at: {}", archive.root)


if __name__ == "__main__":
    main()
