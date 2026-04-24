"""Full-network AEMET monthly scrape — multi-period, resumable.

Walks AEMET's monthly climatological endpoint from 1920 to present for
every station in the network, in small period windows so progress is
checkpointed to GeoParquet at regular intervals. The archive is
idempotent — interrupt with Ctrl-C or kill the process and re-run;
already-fetched rows are not re-requested.

Pacing defaults (see ``_aemet_common.build_archive``) target ~120
req/min globally, well under AEMET's ~150 req/min rolling cap. With
~947 stations × 35 three-year chunks and ~60% chunk hit rate that's
roughly 44k network calls, finishing in ~6 hours of wall time.

Run:
    uv run python scripts/aemet_monthly.py                  # all periods
    uv run python scripts/aemet_monthly.py --start 1980    # resume from 1980
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aemet_common import build_archive, setup_logging


# Five-year periods from the AEMET climatological baseline (1920) to
# the present. Short windows checkpoint progress to disk at frequent
# intervals so an interrupted run loses at most a few hours.
PERIODS: list[tuple[int, int]] = [
    (start, min(start + 4, 2024)) for start in range(1920, 2025, 5)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1920, help="first year to scrape")
    p.add_argument("--end", type=int, default=2024, help="last year to scrape")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("aemet_monthly")
    archive = build_archive("monthly")

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
                "aemet_monthly",
                stations=inventory,
                since=f"{y1}-01-01",
                until=f"{y2}-12-31",
            )
        except KeyboardInterrupt:
            logger.warning(f"interrupted in period {y1}-{y2}; archive is safe")
            raise
        logger.info(
            f"  period {y1}-{y2}: stations={ds.sizes['station']}, "
            f"months={ds.sizes['time']}, "
            f"elapsed={time.monotonic() - period_t0:.1f}s"
        )

    logger.info(f"all periods done in {time.monotonic() - t0:.1f}s")
    logger.info("archive at: {}", archive.root)


if __name__ == "__main__":
    main()
