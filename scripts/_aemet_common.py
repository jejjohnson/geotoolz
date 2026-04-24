"""Shared setup for the AEMET scrape scripts.

Keeps loguru + archive wiring out of the individual scripts so each one
stays a short, readable list of period windows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from xr_toolz.data import AemetArchive, AemetSource


def _default_scratch_root() -> Path:
    """Where the scrape archives live.

    Resolution order (first match wins):

    1. ``AEMET_SCRATCH_ROOT`` environment variable — set this to point
       at any directory, e.g. ``AEMET_SCRATCH_ROOT=~/aemet`` or
       ``/mnt/data/aemet``.
    2. ``XR_TOOLZ_AEMET_ROOT`` environment variable (alias).
    3. ``./scratch/aemet`` under the current working directory — the
       safe portable default that works for any developer / CI.
    """
    for var in ("AEMET_SCRATCH_ROOT", "XR_TOOLZ_AEMET_ROOT"):
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    return Path.cwd() / "scratch" / "aemet"


# Where observations land. The parent script imports + uses this.
SCRATCH_ROOT = _default_scratch_root()

# Where logs go — under the repo, git-ignored.
LOG_ROOT = Path(__file__).resolve().parent.parent / ".logs"


def setup_logging(name: str) -> Path:
    """Configure loguru: stderr + per-script file with default formatting.

    Returns the log-file path for reference (e.g. so tmux users can
    ``tail -f`` it independently of the running process).
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", rotation="20 MB", retention=5)
    return log_path


def build_archive(
    subdir: str,
    *,
    min_interval_s: float = 1.0,
    max_workers: int = 1,
    max_retries: int = 6,
    timeout_s: float = 30.0,
) -> AemetArchive:
    """Build a paced :class:`AemetArchive` pointed at ``SCRATCH_ROOT/subdir``.

    Defaults tuned for long-running scrapes that survive AEMET's
    rate-limit window: **60 req/min** (``min_interval_s=1.0``) with a
    **single worker**. The two-worker / 120 req/min setting we
    originally shipped tripped 429s because the minute bucket never
    actually drained — while one worker was backing off, the other
    kept the bucket hot. ``AemetSource`` now also globally pauses all
    workers on any 429 (see ``_trip_rate_limit``) but the safer
    default is still single-worker at 1 req/s.
    """
    root = SCRATCH_ROOT / subdir
    source = AemetSource(
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_workers=max_workers,
        min_interval_s=min_interval_s,
    )
    archive = AemetArchive(root=root, source=source)
    logger.info(f"archive root: {root}")
    logger.info(
        f"source: max_workers={max_workers}, "
        f"min_interval_s={min_interval_s}, timeout_s={timeout_s}"
    )
    return archive
