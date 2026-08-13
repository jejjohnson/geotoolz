"""xrcore — the xarray-aware operator base for the xrtoolz stack.

Two building blocks, shared by every xarray-facing operator package
(`xrtoolz`, `xrtoolz-einx`, `xrtoolz-sklearn`, …):

- :class:`Operator` — subclasses :class:`pipekit.Operator` and adds
  ``xr.DataTree`` leaf-wise dispatch, so any operator built on it
  handles ``Dataset``, symbolic ``Node``, and ``DataTree`` inputs with
  one ``_apply`` implementation.
- :class:`Signature` — a dict-keyed shape descriptor threaded through
  pipelines for keras-style ``summary()`` tables.
"""

from xrcore.operator import Operator
from xrcore.signature import Signature


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "Operator",
    "Signature",
    "__version__",
]
