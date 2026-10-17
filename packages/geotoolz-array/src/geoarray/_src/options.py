"""Process-wide options: `set_options`, `get_options`, `HostTransferWarning`.

One option today: ``warn_on_transfer``. NumPy code given a JAX carrier
(``np.asarray(g)``) copies the data from the device to the host. That is
correct but slow on a GPU, and silent; with the option on, each copy warns,
which finds the NumPy-only steps of a pipeline that need a JAX kernel.
"""

from __future__ import annotations

import warnings
from types import TracebackType
from typing import Any


__all__ = ["HostTransferWarning", "get_options", "set_options", "warn_host_transfer"]

_OPTIONS: dict[str, Any] = {"warn_on_transfer": False}


class HostTransferWarning(UserWarning):
    """A JAX carrier's data was copied to the host for NumPy code.

    Raised only with ``set_options(warn_on_transfer=True)``. Python's default
    warning filter shows it once per calling line, so each NumPy-only step
    reports once.
    """


def get_options() -> dict[str, Any]:
    """The current options, as a new dict.

    Example:
        >>> get_options()["warn_on_transfer"]
        False
    """
    return dict(_OPTIONS)


class set_options:
    """Set process-wide options, for good or within a ``with`` block.

    Args:
        warn_on_transfer: Warn (`HostTransferWarning`) whenever NumPy code
            copies a JAX carrier's data to the host. ``None`` leaves the
            option as it is.

    Example:
        >>> with set_options(warn_on_transfer=True):
        ...     get_options()["warn_on_transfer"]
        True
        >>> get_options()["warn_on_transfer"]
        False
    """

    def __init__(self, *, warn_on_transfer: bool | None = None) -> None:
        self._previous = dict(_OPTIONS)
        if warn_on_transfer is not None:
            if not isinstance(warn_on_transfer, bool):
                raise TypeError(
                    f"warn_on_transfer must be a bool, got {warn_on_transfer!r}"
                )
            _OPTIONS["warn_on_transfer"] = warn_on_transfer

    def __enter__(self) -> set_options:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _OPTIONS.clear()
        _OPTIONS.update(self._previous)


def warn_host_transfer(what: str, *, stacklevel: int) -> None:
    """Warn that ``what`` copies device data to the host, if asked to."""
    if _OPTIONS["warn_on_transfer"]:
        warnings.warn(
            f"{what} copies a JAX GeoArray's data to the host; use a JAX "
            "kernel to keep it on the device",
            HostTransferWarning,
            stacklevel=stacklevel + 1,
        )
