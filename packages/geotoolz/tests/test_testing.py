"""`geotoolz.testing.check_operator` and the public `geotoolz.carrier` helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _helpers import toy_geotensor
from georeader.geotensor import GeoTensor
from pipekit import Operator

import geotoolz as gz
from geotoolz.carrier import mask_invalid_to_nan, over_frames, wrap_like
from geotoolz.testing import check_operator


def _scene(dtype: Any = np.float32, fill: Any = 0) -> GeoTensor:
    values = np.random.default_rng(0).uniform(1, 100, (3, 6, 6)).astype(dtype)
    return toy_geotensor(
        values,
        fill_value_default=fill,
        attrs={"band_names": ["B02", "B04", "B08"]},
        with_fill_pixels=True,
    )


class ZScore(Operator):
    """The `geotoolz.carrier` docstring example: keeps both contracts."""

    def __init__(self, *, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    @over_frames
    def _apply(self, gt: Any) -> Any:
        values = mask_invalid_to_nan(gt)
        out = (values - self.mean) / self.std
        return wrap_like(gt, out, fill_value_default=np.nan)


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_a_contract_keeping_operator_passes(dtype: Any) -> None:
    scene = _scene(dtype)
    out = check_operator(ZScore(mean=50.0, std=10.0), scene)
    assert out.shape == scene.shape
    assert np.isnan(np.asarray(out)[:, 0, 0]).all()


def test_built_in_operators_pass() -> None:
    check_operator(gz.NDVI(nir="B08", red="B04"), _scene())
    check_operator(gz.DNToReflectance(scale=1e-4), _scene(np.uint16))


class _Positional(ZScore):
    def __init__(self, mean: float, std: float) -> None:  # positional
        super().__init__(mean=mean, std=std)


class _BareArray(ZScore):
    def _apply(self, gt: Any) -> Any:
        return np.asarray(gt, dtype=np.float32)  # drops the carrier


class _SharedAttrs(ZScore):
    def _apply(self, gt: Any) -> Any:
        out = np.asarray(gt, dtype=np.float32) * 2
        return GeoTensor(
            out,
            transform=gt.transform,
            crs=gt.crs,
            fill_value_default=np.nan,
            attrs=gt.attrs,  # the input's dict
        )


class _Mutates(ZScore):
    def _apply(self, gt: Any) -> Any:
        np.asarray(gt)[...] = 0  # writes into the input
        return wrap_like(
            gt, np.asarray(gt, dtype=np.float32), fill_value_default=np.nan
        )


class _IntegerFill(ZScore):
    def _apply(self, gt: Any) -> Any:
        out = np.asarray(gt, dtype=np.float32) / 10  # inherits the integer fill 0
        return wrap_like(gt, out)


class _HoldsAModel(ZScore):
    def __init__(self, *, model: Any = None) -> None:
        super().__init__(mean=0.0, std=1.0)
        self.model = model  # a live object, but not forbid_in_yaml


@pytest.mark.parametrize(
    ("op", "dtype", "match"),
    [
        (_Positional(0.0, 1.0), np.float32, "positionally"),
        (_BareArray(mean=0.0, std=1.0), np.float32, "returned a plain array"),
        (_SharedAttrs(mean=0.0, std=1.0), np.float32, "shares an input's attrs"),
        (_Mutates(mean=0.0, std=1.0), np.float32, "mutated its input"),
        (_IntegerFill(mean=0.0, std=1.0), np.uint16, "expected NaN"),
        (_HoldsAModel(model=object()), np.float32, "forbid_in_yaml"),
    ],
    ids=["positional", "bare-array", "shared-attrs", "mutates", "int-fill", "model"],
)
def test_broken_operators_fail_naming_the_rule(op: Any, dtype: Any, match: str) -> None:
    with pytest.raises((AssertionError, TypeError), match=match):
        check_operator(op, _scene(dtype))


def test_containers_of_operators_round_trip_as_pipekit_refuses_them() -> None:
    # A config holding a *list* of nested operators is refused by pipekit's
    # from_state; that refusal is the expected outcome, not a crash.
    from pipekit import Identity, Sequential

    from geotoolz._src.contract import assert_config_round_trips

    assert_config_round_trips(Sequential([Identity(), Identity()]))


def test_carrier_is_the_shared_toolkit() -> None:
    # The public names are the very objects the built-in operators use.
    from geotoolz._src import shape, valid, wrap

    assert wrap_like is wrap.wrap_like
    assert mask_invalid_to_nan is valid.mask_invalid_to_nan
    assert over_frames is shape.over_frames
