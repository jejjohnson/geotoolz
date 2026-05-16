"""Visualization Operators for RGB composites, display stretching, and overlays."""

from __future__ import annotations

from geotoolz.viz._src.array import (
    blend_rgba,
    composite,
    ensure_rgba,
    gamma_correct_display,
    hillshade,
    rgba_from_categories,
    rgba_from_scalar,
    stretch_to_uint8,
)
from geotoolz.viz._src.operators import (
    AnnotatePoints,
    AnnotatePolygons,
    ApplyColormap,
    ApplyDiscreteColormap,
    FalseColor,
    GammaCorrect,
    Hillshade,
    Overlay,
    ShadedRelief,
    StretchToUint8,
    SWIRComposite,
    ToDisplayRange,
    TrueColor,
)


__all__ = [
    "AnnotatePoints",
    "AnnotatePolygons",
    "ApplyColormap",
    "ApplyDiscreteColormap",
    "FalseColor",
    "GammaCorrect",
    "Hillshade",
    "Overlay",
    "SWIRComposite",
    "ShadedRelief",
    "StretchToUint8",
    "ToDisplayRange",
    "TrueColor",
    "blend_rgba",
    "composite",
    "ensure_rgba",
    "gamma_correct_display",
    "hillshade",
    "rgba_from_categories",
    "rgba_from_scalar",
    "stretch_to_uint8",
]
