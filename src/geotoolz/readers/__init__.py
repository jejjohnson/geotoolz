"""Sensor-specific readers and presets."""

from __future__ import annotations

from geotoolz.readers import toy_sensor
from geotoolz.readers._base import SensorReader, require_optional_dependency


__all__ = ["SensorReader", "require_optional_dependency", "toy_sensor"]
