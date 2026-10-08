"""Install hints for optional extras."""

from __future__ import annotations


def install_hint(extra: str) -> str:
    """The ``pip install`` command for a ``geotoolz-products`` extra.

    Examples:
        >>> install_hint("operators")
        "pip install 'geotoolz-products[operators]'"
    """
    return f"pip install 'geotoolz-products[{extra}]'"


def missing_extra(feature: str, extra: str) -> ImportError:
    """An ``ImportError`` naming the extra that ``feature`` needs."""
    return ImportError(f"{feature} needs the [{extra}] extra: {install_hint(extra)}")
