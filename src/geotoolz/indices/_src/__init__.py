"""Implementation namespace for `geotoolz.indices`.

Two-tier split:

- ``array.py`` — Tier-A pure-numpy primitives (no metadata, no carrier).
  Suitable for unit-testing the math against analytic ground truth.
- ``operators.py`` — Tier-B ``Operator`` wrappers that consume / produce
  ``GeoTensor`` and round-trip ``get_config()`` for Hydra.

Public symbols are re-exported from ``geotoolz.indices``.
"""
