"""Sensor-specific readers and presets."""

from __future__ import annotations

from geotoolz.readers._base import SensorReader, require_optional_dependency


__all__ = ["SensorReader", "modis", "require_optional_dependency", "viirs"]
