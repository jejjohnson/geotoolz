"""Shared setup for the CDS in-situ scrape scripts.

Keeps loguru + archive wiring out of the individual scripts so each
one stays a short, readable period list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from xr_toolz.data import CDSInsituArchive, CDSSource


def _default_scratch_root() -> Path:
    """Where CDS in-situ scrapes land.

    Resolution order (first match wins):

    1. ``CDS_INSITU_SCRATCH_ROOT`` environment variable.
    2. ``XR_TOOLZ_CDS_ROOT`` environment variable (alias).
    3. ``~/cloudfiles/code/Users/adm.jjohnson72/scratch/cds_insitu`` if
       that path exists on the machine (matches the shared Azure VM
       layout used for long-running scrapes).
    4. ``./scratch/cds_insitu`` under the CWD — portable last resort.
       The repo's ``.gitignore`` excludes ``scratch/`` so downloaded
       data never lands in git even if this default fires inside a
       checkout.
    """
    for var in ("CDS_INSITU_SCRATCH_ROOT", "XR_TOOLZ_CDS_ROOT"):
        override = os.environ.get(var)
        if override:
            return Path(override).expanduser()
    shared = Path("/home/azureuser/cloudfiles/code/Users/adm.jjohnson72/scratch")
    if shared.is_dir():
        return shared / "cds_insitu"
    return Path.cwd() / "scratch" / "cds_insitu"


SCRATCH_ROOT = _default_scratch_root()
LOG_ROOT = Path(__file__).resolve().parent.parent / ".logs"


def setup_logging(name: str) -> Path:
    """Configure loguru: stderr + per-script rotating file log.

    Returns the log-file path so tmux users can ``tail -f`` it.
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", rotation="20 MB", retention=5)
    return log_path


def build_archive(
    preset: str,
    *,
    subdir: str | None = None,
    time_aggregation: str = "daily",
) -> CDSInsituArchive:
    """Build a :class:`CDSInsituArchive` rooted at ``SCRATCH_ROOT/subdir``.

    Args:
        preset: ``"cds_insitu_land"`` or ``"cds_insitu_marine"``.
        subdir: Sub-directory under :data:`SCRATCH_ROOT`. Defaults to
            ``preset_time_aggregation``, so each preset/tier combo gets
            its own tree without stepping on neighbours.
        time_aggregation: ``"sub_daily"`` / ``"daily"`` / ``"monthly"``.
            Land respects this on every request; marine ignores it
            (marine ships a single aggregation tier baked in).
    """
    root = SCRATCH_ROOT / (subdir or f"{preset}_{time_aggregation}")
    source = CDSSource()
    archive = CDSInsituArchive(
        root=root,
        preset=preset,
        source=source,
        time_aggregation=time_aggregation,
    )
    logger.info(f"archive root: {root}")
    logger.info(f"source: preset={preset} time_aggregation={time_aggregation}")
    return archive
