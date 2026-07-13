"""Patcher-layer exception types.

Kept minimal — most patcher misuse raises `ValueError`/`TypeError`. Bespoke
exceptions land here when the same failure mode is raised from multiple
modules and benefits from `except SpecificError:` filtering.
"""

from __future__ import annotations


class IncompleteScanConfiguration(Exception):
    """A sampler's ``(size, step)`` does not exactly tile the domain.

    Raised by `SpatialRegularStride(check_full_scan=True)` (and any
    follow-up sampler that opts into the same strict-tiling contract).
    The temporal counterpart raises a plain `ValueError` via
    `divide_evenly` because the tiling check is part of the math, not a
    sampler-level config; both serve the same robustness role.

    Mirrors `xrpatcher`'s exception of the same name so migration paths
    can keep their existing ``except IncompleteScanConfiguration:``
    clauses unchanged.
    """


__all__ = ["IncompleteScanConfiguration"]
