"""Patcher-layer exception types.

Kept minimal — most patcher misuse raises `ValueError`/`TypeError`. Bespoke
exceptions land here when the same failure mode is raised from multiple
modules and benefits from `except SpecificError:` filtering.
"""

from __future__ import annotations


class IncompleteScanConfiguration(ValueError):
    """A sampler's ``(size, step)`` does not exactly tile the domain.

    Raised by `spatial.sampler.RegularStride(check_full_scan=True)` and by
    `TemporalPatcher` when its `temporal.sampler.RegularStride(check_full_scan=True)`
    windows leave time steps uncovered. A `ValueError`, so a plain
    ``except ValueError:`` catches it too.

    Mirrors `xrpatcher`'s exception of the same name so migration paths
    can keep their existing ``except IncompleteScanConfiguration:``
    clauses unchanged.
    """


__all__ = ["IncompleteScanConfiguration"]
