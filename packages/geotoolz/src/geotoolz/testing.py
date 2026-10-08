"""Check that an operator keeps the stack's contracts.

`check_operator` runs, on one operator and one sample scene, the checks
geotoolz runs on every built-in operator (``tests/test_operator_contract.py``):
the pipekit `Operator` contract (keyword-only constructor, JSON config that
round-trips, graph mode) and the `GeoTensor` carrier contract (same carrier
out, fresh and consistent ``attrs``, a fill that suits the output, the input
left untouched, ``(T, C, H, W)`` stacks handled or rejected). Use it in the
tests of an operator you write on top of the stack.

Example:
    ::

        from geotoolz.testing import check_operator

        def test_zscore(scene):  # scene: GeoTensor (C, H, W) with some nodata
            out = check_operator(ZScore(mean=0.3, std=0.1), scene)
            assert out.shape == scene.shape
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geotoolz._src import contract


__all__ = ["check_operator"]


def check_operator(op: Any, sample: Any, *others: Any, gap_filler: bool = False) -> Any:
    """Assert that ``op`` keeps the pipekit `Operator` and `GeoTensor` contracts.

    Args:
        op: The operator instance to check.
        sample: A `GeoTensor` scene ``op`` accepts — ideally ``(C, H, W)``
            with ``band_names`` in its ``attrs`` and a few nodata pixels, so
            the attrs and nodata checks have something to check.
        *others: Further carriers of a multi-input operator, on ``sample``'s
            grid.
        gap_filler: ``True`` for an operator that fills nodata by design;
            the input's nodata pixels may then become valid.

    Returns:
        ``op(sample, *others)``, for further assertions.

    Raises:
        AssertionError: Naming the first broken rule.
    """
    cls = type(op)
    contract.assert_keyword_only(cls)
    contract.assert_config_round_trips(op)
    contract.assert_graph_mode(op, 1 + len(others))

    before = np.array(sample, copy=True)
    attrs_before = dict(getattr(sample, "attrs", {}) or {})
    out = op(sample, *others)
    np.testing.assert_array_equal(
        np.asarray(sample), before, err_msg=f"{cls.__name__} mutated its input"
    )
    assert dict(getattr(sample, "attrs", {}) or {}) == attrs_before, (
        f"{cls.__name__} mutated its input's attrs"
    )
    if cls._terminal:
        return out
    contract.assert_same_carrier(sample, out)
    contract.assert_fresh_consistent_attrs(sample, out)
    for gt in contract.geotensors(out):
        contract.assert_fill_matches_dtype(sample, gt, gap_filler=gap_filler)

    try:
        plain = op(np.asarray(sample), *(np.asarray(o) for o in others))
    except (TypeError, ValueError):
        pass  # needs georeferencing or band names: a documented rejection
    else:
        contract.assert_same_carrier(np.asarray(sample), plain)

    if not others and np.ndim(sample) in (2, 3):
        contract.assert_time_stack(op, sample)
    return out
