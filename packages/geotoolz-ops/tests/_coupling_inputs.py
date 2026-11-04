"""Probe inputs for the coupling check: which toy input each operator gets.

Shared by ``test_operator_coupling.py`` and ``scripts/operator_coupling.py``.
It reads the operator contract test's tables (constructor kwargs, second
carriers, primary inputs, N-ary inputs) and never changes them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import test_operator_contract as tc
from _helpers import toy_geoarray

from geoops.testing import Measured, measure_coupling, measure_frames


#: Probe-only constructor settings, for operators the contract test builds
#: but that cannot run on a toy scene as built.
PROBE_KWARGS: dict[str, dict[str, Any]] = {
    **{
        f"normalize._src.operators.{name}": {"fit_on_call": True}
        for name in ("StandardScaler", "MinMaxScaler", "RobustScaler")
    },
    **{
        f"qa._src.operators.{name}": {"qa_band": 0, "bits": [3]}
        for name in (
            "MaskCirrus",
            "MaskCloudShadow",
            "MaskClouds",
            "MaskSnow",
            "MaskWater",
        )
    },
    "geom._src.operators.CropToBounds": {
        "bounds": (500_030.0, 3_999_870.0, 500_130.0, 3_999_970.0)
    },
}

#: Operators that take one list of carriers, probed like the N-ary reducers.
PROBE_LISTS: tuple[str, ...] = (
    "geom._src.operators.Mosaic",
    "spectral._src.operators.StackBands",
)

_LABEL_OPERATOR = "measure._src.operators.LabelConnectedComponents"


def build_for_probe(cls: type) -> Any:
    """``cls`` built as the contract test builds it, plus `PROBE_KWARGS`."""
    key = tc._key(cls)
    if key not in PROBE_KWARGS:
        return tc._build_seeded(cls)
    base = tc.CTOR_KWARGS.get(key, {})
    kwargs = {**(base() if callable(base) else base), **PROBE_KWARGS[key]}
    if "seed" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("seed", 0)
    return cls(**kwargs)


def domain_inputs(key: str) -> list[Any]:
    """Inputs an operator family really reads, tried before the generic ones."""
    scene = tc._scene()
    if key.startswith("mask.") or key == _LABEL_OPERATOR:
        return [tc._mask_like(scene)]
    if key.startswith("qa.") and "S2" not in key and "Landsat" not in key:
        qa = np.random.default_rng(0).integers(0, 2**12, (16, 16)).astype(np.uint16)
        return [toy_geoarray(qa, fill_value=None)]
    if key.startswith("restore.") and ("GapFill" in key or "Outlier" in key):
        values = np.asarray(scene).copy()
        if "Outlier" in key:
            values[:, 8, 9] = 50.0  # an outlier next to the centre pixel
        else:
            values[:, 8:10, 9:11] = np.nan  # a hole next to the centre pixel
        return [scene.replace(data=values, fill_value=np.nan)]
    return []


def _frames(key: str) -> tuple[list[Any], bool] | None:
    if key in PROBE_LISTS:
        return tc._two_frames(), True
    if key in tc.NARY_INPUTS:
        given = tc.NARY_INPUTS[key]()
        as_list = len(given) == 1 and isinstance(given[0], list)
        return (given[0] if as_list else given), as_list
    return None


def probe(cls: type) -> tuple[Measured | None, str]:
    """``(measurement, why)``: the measurement, or why there is none."""
    key = tc._key(cls)

    def build() -> Any:
        return build_for_probe(cls)

    try:
        op = build()
    except (Exception, pytest.skip.Exception) as exc:
        return None, f"cannot build: {exc}"[:160]
    frames = _frames(key)
    if frames is not None:
        return measure_frames(build, frames[0], as_list=frames[1]), ""
    if not tc._single_carrier_in(op):
        return None, "no single primary input"
    primary = tc.GRAPH_PRIMARY.get(key)
    candidates = [
        *([primary()] if primary is not None else []),
        *domain_inputs(key),
        *tc._toy_inputs(),
        *tc._attributed_inputs(),
    ]
    why = "no toy input accepted"
    for x in candidates:
        if not tc._is_carrier(x) or np.ndim(x) < 2:
            continue
        try:
            tc._call(build(), x)
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            continue
        return measure_coupling(build, x, call=tc._call), ""
    return None, why[:160]


def builder(cls: type) -> Callable[[], Any]:
    """A zero-argument builder for ``cls``, for `measure_coupling`."""
    return lambda: build_for_probe(cls)
