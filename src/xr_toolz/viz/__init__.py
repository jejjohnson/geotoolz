"""Visualization operators for xr_toolz.

Per D10, viz operators are first-class :class:`Operator` instances
that return :class:`matplotlib.figure.Figure` (or
``(Figure, Axes)``). They are *terminal* — a non-Dataset return must
appear only as the last step of a :class:`Sequential` or as one of
the leaves of a :class:`Graph`.

Submodules:

- :mod:`xr_toolz.viz.validation` — V6 validation panels keyed to
  the V1–V5 metric outputs.
"""
