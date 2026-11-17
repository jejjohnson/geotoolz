"""Errors raised by the local product openers, named by what failed."""

from __future__ import annotations


class ProductGridError(ValueError):
    """A companion raster is not on the grid of the scene it belongs to.

    Raised when a sensitivity or uncertainty file has another ``(y, x)``
    shape than the enhancement, or coordinates that differ from it by
    more than the tolerance. Nothing is resampled to make them agree.

    Example:
        >>> from xrproducts import ProductGridError
        >>> issubclass(ProductGridError, ValueError)
        True
    """


class MissingPropertyError(KeyError):
    """A plume-complex feature lacks a property the opener needs.

    Example:
        >>> from xrproducts import MissingPropertyError
        >>> issubclass(MissingPropertyError, KeyError)
        True
    """
