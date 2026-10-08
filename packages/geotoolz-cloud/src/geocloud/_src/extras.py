"""One spelling for "install the extra" hints.

The import name is ``geocloud`` but the distribution is ``geotoolz-cloud``,
so every missing-extra message is built here rather than spelled by hand.
"""

from __future__ import annotations


DISTRIBUTION = "geotoolz-cloud"


def install_hint(extra: str) -> str:
    """The pip command that installs ``extra``.

    Examples:
        >>> install_hint("cog")
        "pip install 'geotoolz-cloud[cog]'"
    """
    return f"pip install '{DISTRIBUTION}[{extra}]'"


def missing_extra(feature: str, extra: str) -> ImportError:
    """``ImportError`` raised when ``feature`` needs an uninstalled ``extra``.

    Examples:
        >>> "geotoolz-cloud[cog]" in str(missing_extra("CogSource", "cog"))
        True
    """
    return ImportError(
        f"`{feature}` requires the `{extra}` extra. "
        f"Install with `{install_hint(extra)}`."
    )
