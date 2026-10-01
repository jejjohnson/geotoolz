"""Sensor-specific readers and presets."""

from __future__ import annotations

from geotoolz.readers import toy_sensor
from geotoolz.readers._src.base import SensorReader


__all__ = ["SensorReader", "toy_sensor"]
