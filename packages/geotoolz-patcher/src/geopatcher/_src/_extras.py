"""One spelling for "install the extra" hints.

The import name is ``geopatcher`` but the distribution is
``geotoolz-patcher``, so every missing-extra message is built here rather
than spelled by hand at each guard.
"""

from __future__ import annotations


DISTRIBUTION = "geotoolz-patcher"


def install_hint(extra: str) -> str:
    """The pip command that installs ``extra``.

    Examples:
        >>> install_hint("grid")
        "pip install 'geotoolz-patcher[grid]'"
        >>> install_hint("cog")
        "pip install 'geotoolz-patcher[cog]'"
    """
    return f"pip install '{DISTRIBUTION}[{extra}]'"


def missing_extra(feature: str, extra: str, pip_pkgs: str | None = None) -> ImportError:
    """``ImportError`` raised when ``feature`` needs an uninstalled ``extra``.

    Examples:
        >>> err = missing_extra("XarrayField", "grid", "xarray>=2024.1")
        >>> "pip install 'geotoolz-patcher[grid]'" in str(err)
        True
        >>> "pip install xarray>=2024.1" in str(err)
        True
    """
    alternative = f" (or `pip install {pip_pkgs}`)" if pip_pkgs else ""
    return ImportError(
        f"`{feature}` requires the `{extra}` extra. "
        f"Install with `{install_hint(extra)}`{alternative}."
    )
