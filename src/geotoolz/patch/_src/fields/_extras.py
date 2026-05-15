"""Friendly error for missing optional-extra backend libraries."""

from __future__ import annotations


def _missing_extra(backend: str, extra: str, pip_pkgs: str) -> ImportError:
    return ImportError(
        f"`{backend}` requires the `{extra}` extra. "
        f"Install with `pip install 'geotoolz[{extra}]'` "
        f"(or `pip install {pip_pkgs}`)."
    )
