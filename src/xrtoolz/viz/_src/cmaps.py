"""Colormap resolution for spatial viz panels.

Looks up a sensible default colormap for a given canonical variable
name (or :class:`xrreader.types.Variable` instance) by reading the
``cmap`` field on entries of the curated
:data:`xrreader.types.REGISTRY`. Falls back to ``default`` (default
``"viridis"``) when the variable is unknown or the registry entry
has no ``cmap`` set.
"""

from __future__ import annotations

from xrreader.types import REGISTRY, Variable


def cmap_for(variable: str | Variable | None, default: str = "viridis") -> str:
    """Return the registry-recommended colormap for ``variable``.

    Args:
        variable: Variable instance, canonical short name (case-insensitive),
            or ``None``.
        default: Fallback colormap when ``variable`` is ``None``, unknown, or
            its registry entry has no ``cmap``.

    Returns:
        Matplotlib colormap name (or any string the caller accepted on
        registration — cmocean colormaps via ``"cmo.<name>"`` work too
        when ``cmocean`` is installed and registered).
    """
    if variable is None:
        return default
    if isinstance(variable, Variable):
        return variable.cmap or default
    entry = REGISTRY.get(variable) or REGISTRY.get(variable.lower())
    if entry is None or entry.cmap is None:
        return default
    return entry.cmap


__all__ = ["cmap_for"]
