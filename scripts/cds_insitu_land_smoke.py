"""Smoke test for the CDS in-situ surface-land adapter.

Downloads one year of daily observations and verifies the archive
round-trips through GeoParquet. Intended to run under a few minutes
(the CDS queue permitting).

Run:

    # Ensure credentials are set in .env or env vars:
    #   CDSAPI_URL=https://cds.climate.copernicus.eu/api
    #   CDSAPI_KEY=<uid>:<api-key>
    uv run python scripts/cds_insitu_land_smoke.py
"""

from __future__ import annotations

from _cds_insitu_common import build_archive, setup_logging
from loguru import logger


def main() -> None:
    log_path = setup_logging("cds_insitu_land_smoke")
    logger.info(f"logging to {log_path}")

    archive = build_archive(
        preset="cds_insitu_land",
        subdir="smoke_land_daily",
        time_aggregation="daily",
    )

    # One-year smoke pull — keeps the request small and reviewable.
    logger.info("starting smoke sync: 2020 daily land observations")
    fresh = archive.sync("2020-01-01", "2020-12-31")
    logger.info(f"fetched {len(fresh)} rows")

    gdf = archive.load()
    logger.info(
        f"archive now holds {len(gdf)} rows across "
        f"{gdf['station_id'].nunique()} stations"
    )
    logger.info("smoke test OK")


if __name__ == "__main__":
    main()
