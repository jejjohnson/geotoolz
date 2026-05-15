"""Composition operators — Fanout.

`Fanout` is sugar over `Graph` for the common case of "one input, many
named outputs" — useful when you want N derived products from one
read instead of running N independent `Sequential`s. The input
GeoTensor flows into each branch unchanged; the outputs are returned
keyed by branch name.

See `tips_n_tricks.md` §"Fanout".
"""

from __future__ import annotations

from typing import Any

from geotoolz.core._src.operator import Carrier, Operator


class Fanout(Operator):
    """One input → dict of outputs (sugar over `Graph`).

    Each branch is applied to the same input GeoTensor; the outputs are
    returned as a dict keyed by the branch name.

    Args:
        branches: Map of ``output-name → Operator``. Each operator
            receives the same input independently and contributes one
            entry to the returned dict.

    Examples:
        Compute three indices from one scene with one read::

            products = Fanout({
                "ndvi": NDVI(red_idx=2, nir_idx=3),
                "ndwi": NDWI(green_idx=1, nir_idx=3),
                "rgb":  S2_L2A_RGB(),
            })(gt)
            # {"ndvi": GeoTensor, "ndwi": GeoTensor, "rgb": GeoTensor}

        Equivalent `Graph` form (more verbose, identical result)::

            img = Input("image")
            g = Graph(
                inputs={"image": img},
                outputs={
                    "ndvi": NDVI(red_idx=2, nir_idx=3)(img),
                    "ndwi": NDWI(green_idx=1, nir_idx=3)(img),
                    "rgb":  S2_L2A_RGB()(img),
                },
            )

    Raises:
        TypeError: if any branch is not an `Operator`, or if no branches
            are provided.
    """

    def __init__(self, branches: dict[str, Operator]) -> None:
        if not branches:
            raise TypeError("Fanout requires at least one branch.")
        for name, op in branches.items():
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Fanout branch {name!r} is {type(op).__name__}, expected Operator."
                )
        self.branches = dict(branches)

    def _apply(self, gt: Carrier) -> dict[str, Any]:
        return {name: op(gt) for name, op in self.branches.items()}

    def get_config(self) -> dict[str, Any]:
        return {
            "branches": {
                name: {
                    "class": type(op).__name__,
                    "config": op.get_config(),
                }
                for name, op in self.branches.items()
            }
        }
