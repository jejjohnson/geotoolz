"""Colormap resolution for spatial viz panels.

Looks up a sensible default colormap for a given canonical variable
name (or :class:`xrtoolz.types.Variable` instance) by reading the
``cmap`` field on entries of the curated
:data:`xrtoolz.types.REGISTRY`. Falls back to ``default`` (default
``"viridis"``) when the variable is unknown or the registry entry
has no ``cmap`` set.
"""

from __future__ import annotations

from xrtoolz.types._src.variable import REGISTRY, Variable


def cmap_for(var: str | Variable | None, default: str = "viridis") -> str:
    """Return the registry-recommended colormap for ``var``.

    Args:
        var: Variable instance, canonical short name (case-insensitive),
            or ``None``.
        default: Fallback colormap when ``var`` is ``None``, unknown, or
            its registry entry has no ``cmap``.

    Returns:
        Matplotlib colormap name (or any string the caller accepted on
        registration — cmocean colormaps via ``"cmo.<name>"`` work too
        when ``cmocean`` is installed and registered).
    """
    if var is None:
        return default
    if isinstance(var, Variable):
        return var.cmap or default
    entry = REGISTRY.get(var) or REGISTRY.get(var.lower())
    if entry is None or entry.cmap is None:
        return default
    return entry.cmap


__all__ = ["cmap_for"]
