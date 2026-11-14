"""Cross-cutting utilities — domain-agnostic helpers reused across modules.

Hosts the internal finite-mask, grid-spacing, validation, and
optional-import helpers under :mod:`xrops.utils._src`, and the public
`pack_dataset` / `unpack_dataset`, which stack a Dataset's variables along
a ``variable`` dim and back, plus the ``ReduceDims`` operator.

scikit-learn on xarray is :mod:`xrops.learn`.
"""

from xrops.utils._src.dataset import pack_dataset, unpack_dataset
from xrops.utils.operators import ReduceDims


__all__ = ["ReduceDims", "pack_dataset", "unpack_dataset"]
