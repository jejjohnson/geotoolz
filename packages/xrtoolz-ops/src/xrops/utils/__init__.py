"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Hosts the internal finite-mask, grid-spacing, validation, and
optional-import helpers under :mod:`xrops.utils._src`, and the public
`pack_dataset` / `unpack_dataset`, which stack a Dataset's variables along
a ``variable`` dim and back.

scikit-learn on xarray is :mod:`xrops.learn`.
"""

from xrops.utils._src.dataset import pack_dataset, unpack_dataset


__all__ = ["pack_dataset", "unpack_dataset"]
