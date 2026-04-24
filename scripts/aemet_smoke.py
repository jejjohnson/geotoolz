"""AEMET smoke-test script — validates the full pipeline (~5-10 min).

What it does:

1. Sets up the paced archive (rate-limited at 120 req/min).
2. Refreshes the station inventory.
3. Fetches twenty years of monthly data (2005-2024) for about two
   stations per community (≈37-40 stations). That's ~7 three-year
   chunks × ~40 stations × 2 hops ≈ 550 network calls, finishing in
   5-10 minutes at the default 0.5 s global pace.
4. Reads the archive back as a GeoDataFrame and prints a brief
   summary.

The scope is deliberately wide enough to exercise the pacing gate,
chunk-stitching, archive append, and GeoParquet round-trip under
sustained load — not just a one-shot validation. If this runs clean
you're safe to launch the long monthly / daily scrapers.

Everything writes to ``scratch/aemet/smoke/`` and logs to
``xr_toolz/.logs/aemet_smoke.log``. Safe to interrupt and re-run.

Run:
    uv run python scripts/aemet_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

from xr_toolz.types import StationCollection


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aemet_common import build_archive, setup_logging


def main() -> None:
    setup_logging("aemet_smoke")
    archive = build_archive("smoke")

    t0 = time.monotonic()
    logger.info("refreshing station inventory")
    inventory = archive.sync_stations()
    logger.info(
        f"inventory: {len(inventory)} stations across "
        f"{len(inventory.communities())} communities"
    )

    # Pick ~2 well-instrumented reference stations per community so
    # the smoke covers every autonomous community and exercises the
    # pacing / retry / merge paths under sustained load.
    per_community = 2
    picks: list = []
    for community in inventory.communities():
        pool = inventory.filter(community=community, has_wmo=True)
        if len(pool) < per_community:
            pool = inventory.filter(community=community)
        picks.extend(list(pool)[:per_community])
    sample = StationCollection.from_iter(picks)
    logger.info(
        f"sampled {len(sample)} stations across {len(sample.communities())} communities"
    )

    logger.info("fetching monthly 2005-2024 for the sample (~5-10 min expected)")
    ds = archive.sync(
        "aemet_monthly",
        stations=sample,
        since="2005-01-01",
        until="2024-12-31",
    )
    logger.info(
        f"fetched slice: stations={ds.sizes['station']}, "
        f"months={ds.sizes['time']}, variables={len(ds.data_vars)}"
    )

    logger.info("reading archive back as GeoParquet")
    gdf = archive.load("aemet_monthly")
    logger.info(f"archive rows: {len(gdf):,}")
    logger.info(f"archive CRS:  EPSG:{gdf.crs.to_epsg()}")
    non_null = gdf["air_temperature_daily_mean"].notna().sum()
    logger.info(
        f"air_temperature_daily_mean non-null: {non_null}/{len(gdf)} "
        f"({non_null / len(gdf):.0%})"
    )

    logger.info(f"done in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
