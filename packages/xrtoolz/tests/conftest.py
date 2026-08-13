"""Shared pytest configuration.

Tier markers (``slow`` / ``integration``) are registered in
``pyproject.toml``; the automatic CI run excludes both. Dask-backed
parity coverage lives in ``tests/test_dask_compat.py`` under the
``dask`` marker.
"""

from __future__ import annotations
