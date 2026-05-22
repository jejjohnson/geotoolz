"""Shared spatial helpers for interpolation primitives."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.neighbors import BallTree, KDTree


Metric = Literal["euclidean", "haversine"]
BandwidthRule = Literal["scott", "silverman"]


def _build_tree(src_xy: np.ndarray, metric: Metric) -> Any:
    """Build the sklearn neighbour tree appropriate for ``metric``."""
    if metric == "haversine":
        return BallTree(src_xy, metric="haversine")
    return KDTree(src_xy, metric="euclidean")


def _to_metric_xy(lons: np.ndarray, lats: np.ndarray, metric: Metric) -> np.ndarray:
    """Convert lon/lat arrays to the coordinate order expected by sklearn trees."""
    if metric == "haversine":
        return np.deg2rad(np.column_stack([lats, lons]))
    return np.column_stack([lons, lats])


def _bandwidth_rule(pts: np.ndarray, rule: BandwidthRule) -> float:
    """Return Scott or Silverman bandwidth from mean coordinate variance."""
    n, d = pts.shape
    coordinate_variance = np.var(pts, axis=0, ddof=1)
    sigma = float(np.sqrt(np.mean(coordinate_variance)))
    if rule == "scott":
        bandwidth = n ** (-1.0 / (d + 4)) * sigma
    elif rule == "silverman":
        bandwidth = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4)) * sigma
    else:
        raise ValueError(f"unknown bandwidth rule {rule!r}")
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"bandwidth rule {rule!r} produced {bandwidth!r}")
    return bandwidth


__all__ = ["BandwidthRule", "Metric", "_bandwidth_rule", "_build_tree", "_to_metric_xy"]
