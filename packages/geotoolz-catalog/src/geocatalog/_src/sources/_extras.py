"""Friendly errors for missing optional-extra source-adapter libraries."""

from __future__ import annotations


def _missing_extra(adapter: str, extra: str, pip_pkgs: str) -> ImportError:
    """Build an ImportError that tells the user exactly how to fix it.

    Mirrors the ``_missing_extra`` helper in
    ``geopatcher._src.fields._extras`` so the message style is
    consistent across the project.
    """
    return ImportError(
        f"`{adapter}` requires the `{extra}` extra. "
        f"Install with `pip install 'geocatalog[{extra}]'` "
        f"(or `pip install {pip_pkgs}`)."
    )
