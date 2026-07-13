"""Remote-sensing-safe data augmentations.

The operators in this module preserve ``GeoTensor`` metadata, avoid
band-channel shuffling by default, and centralise per-operator plus
per-call seed handling for reproducible training pipelines.
"""

from __future__ import annotations

from geotoolz.augment._src.operators import (
    AtmosphericHaze,
    BandDropout,
    BandJitter,
    BrightnessJitter,
    Compose,
    ContrastJitter,
    CutMix,
    GaussianNoise,
    RandomCrop,
    RandomFlip,
    RandomRotate90,
    RandomShift,
    SimulatedClouds,
    SpeckleNoise,
    SunAngleJitter,
)


__all__ = [
    "AtmosphericHaze",
    "BandDropout",
    "BandJitter",
    "BrightnessJitter",
    "Compose",
    "ContrastJitter",
    "CutMix",
    "GaussianNoise",
    "RandomCrop",
    "RandomFlip",
    "RandomRotate90",
    "RandomShift",
    "SimulatedClouds",
    "SpeckleNoise",
    "SunAngleJitter",
]
