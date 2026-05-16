"""VIIRS reader skeleton with optional dependency guard."""

from __future__ import annotations

from pathlib import Path

from geotoolz.readers._base import require_optional_dependency


class Reader:
    """VIIRS reader entry point.

    Args:
        path: VIIRS scene path.

    Examples:
        >>> from geotoolz.readers.viirs import Reader
        >>> Reader("scene.h5")  # doctest: +SKIP
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        require_optional_dependency("h5py", extra="viirs")
        raise NotImplementedError(
            "VIIRS reader parsing is sensor-specific future work."
        )
