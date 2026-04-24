"""Multi-year CDS in-situ surface-land scrape (resumable, tmux-friendly).

Scrapes daily observations from ``START_YEAR`` to the current year,
one year at a time. The archive's manifest tracks completed chunks,
so re-running resumes from where it left off.

Run under tmux:

    tmux new -s cds_land
    uv run python scripts/cds_insitu_land_full.py
    # detach: Ctrl-b d

Logs rotate at 20 MB under ``.logs/cds_insitu_land_full.log``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _cds_insitu_common import build_archive, setup_logging
from loguru import logger


START_YEAR = 1950


def main() -> None:
    log_path = setup_logging("cds_insitu_land_full")
    logger.info(f"logging to {log_path}")

    archive = build_archive(
        preset="cds_insitu_land",
        subdir="full_land_daily",
        time_aggregation="daily",
    )

    end_year = datetime.now(UTC).year
    logger.info(f"starting full sync: {START_YEAR}..{end_year} daily land")
    archive.sync(f"{START_YEAR}-01-01", f"{end_year}-12-31")

    gdf = archive.load()
    logger.info(
        f"archive now holds {len(gdf)} rows across "
        f"{gdf['station_id'].nunique()} stations"
    )


if __name__ == "__main__":
    main()
