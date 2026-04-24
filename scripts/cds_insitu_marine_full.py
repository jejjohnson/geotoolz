"""Multi-year CDS in-situ surface-marine scrape (resumable, tmux-friendly).

Scrapes daily marine-surface observations from ``START_YEAR`` to the
current year, one year at a time. Resumes from the manifest on re-run.

Run under tmux:

    tmux new -s cds_marine
    uv run python scripts/cds_insitu_marine_full.py
    # detach: Ctrl-b d
"""

from __future__ import annotations

from datetime import UTC, datetime

from _cds_insitu_common import build_archive, setup_logging
from loguru import logger


START_YEAR = 1978


def main() -> None:
    log_path = setup_logging("cds_insitu_marine_full")
    logger.info(f"logging to {log_path}")

    archive = build_archive(
        preset="cds_insitu_marine",
        subdir="full_marine_daily",
        time_aggregation="daily",
    )

    end_year = datetime.now(UTC).year
    logger.info(f"starting full sync: {START_YEAR}..{end_year} daily marine")
    archive.sync(f"{START_YEAR}-01-01", f"{end_year}-12-31")

    gdf = archive.load()
    logger.info(
        f"archive now holds {len(gdf)} rows across "
        f"{gdf['station_id'].nunique()} platforms"
    )


if __name__ == "__main__":
    main()
