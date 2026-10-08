"""Fixtures for the Himawari AHI tests.

The HSD reader needs only numpy, so its tests always run; the L2 reader
tests need the ``[himawari]`` extra (h5py) and are left uncollected on a
slim install.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest


collect_ignore_glob = (
    ["test_himawari_l2.py", "test_himawari_integration.py"]
    if importlib.util.find_spec("h5py") is None
    else []
)


@pytest.fixture
def hsd(tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``hsd(counts, name=..., **write_hsd_kwargs)`` → segment path."""

    def make(counts: np.ndarray, *, name: str | None = None, **kwargs: Any) -> Path:
        from _hsd import segment_name, write_hsd

        if name is None:
            name = segment_name(
                kwargs.get("band", 13),
                kwargs.get("segment", 1),
                kwargs.get("total_segments", 1),
                area=kwargs.get("area", "FLDK"),
            )
        return write_hsd(tmp_path / name, counts, **kwargs)

    return make


@pytest.fixture
def full_disk(hsd: Callable[..., Path]) -> Callable[..., list[Path]]:
    """Factory: the 100 x 100 disk of ``band`` as ``segments`` HSD files.

    Counts rise along each row (``1000 + column``) so placements are
    checkable; returns the paths in segment order.
    """
    from _hsd import SIZE, segment_name

    def make(
        band: int = 13, segments: int = 4, *, bz2: bool = False, **kw: Any
    ) -> list[Path]:
        lines = SIZE // segments
        row = 1000 + np.arange(SIZE, dtype=np.uint16)
        return [
            hsd(
                np.tile(row, (lines, 1)) + np.uint16(100 * s),
                name=segment_name(band, s, segments) + (".bz2" if bz2 else ""),
                band=band,
                segment=s,
                total_segments=segments,
                **kw,
            )
            for s in range(1, segments + 1)
        ]

    return make
