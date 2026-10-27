"""`geoops.testing.check_operator` and the public `geoops.carrier` helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _helpers import toy_geoarray
from geoarray import GeoArray
from pipekit import Operator

import geoops as gz
from geoops.carrier import mask_invalid_to_nan, over_frames, wrap_like
from geoops.testing import check_operator


def _scene(dtype: Any = np.float32, fill: Any = 0) -> GeoArray:
    values = np.random.default_rng(0).uniform(1, 100, (3, 6, 6)).astype(dtype)
    return toy_geoarray(
        values,
        fill_value=fill,
        attrs={"band_names": ["B02", "B04", "B08"]},
        with_fill_pixels=True,
    )


class ZScore(Operator):
    """The `geoops.carrier` docstring example: keeps both contracts."""

    backends = ("numpy",)

    def __init__(self, *, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    @over_frames
    def _apply(self, gt: Any) -> Any:
        values = mask_invalid_to_nan(gt)
        out = (values - self.mean) / self.std
        return wrap_like(gt, out, fill_value=np.nan)


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


class _BuildsByHand(ZScore):
    def _apply(self, gt: Any) -> Any:
        out = np.asarray(gt, dtype=np.float32) * 2
        out[..., np.asarray(gt)[0] == getattr(gt, "fill_value", None)] = np.nan
        if not hasattr(gt, "transform"):
            return out
        # Built by hand instead of wrap_like: a georeader GeoTensor input is
        # silently accepted and comes back as a GeoArray.
        return GeoArray(
            out,
            transform=gt.transform,
            crs=gt.crs,
            fill_value=np.nan,
            attrs=gt.attrs,
        )


class _Mutates(ZScore):
    def _apply(self, gt: Any) -> Any:
        np.asarray(gt)[...] = 0  # writes into the input
        return wrap_like(gt, np.asarray(gt, dtype=np.float32), fill_value=np.nan)


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
        (
            _BuildsByHand(mean=0.0, std=1.0),
            np.float32,
            "accepted a georeader GeoTensor",
        ),
        (_Mutates(mean=0.0, std=1.0), np.float32, "mutated its input"),
        (_IntegerFill(mean=0.0, std=1.0), np.uint16, "expected NaN"),
        (_HoldsAModel(model=object()), np.float32, "forbid_in_yaml"),
    ],
    ids=[
        "positional",
        "bare-array",
        "accepts-geotensor",
        "mutates",
        "int-fill",
        "model",
    ],
)
def test_broken_operators_fail_naming_the_rule(op: Any, dtype: Any, match: str) -> None:
    with pytest.raises((AssertionError, TypeError), match=match):
        check_operator(op, _scene(dtype))


def test_containers_of_operators_round_trip_as_pipekit_refuses_them() -> None:
    # A config holding a *list* of nested operators is refused by pipekit's
    # from_state; that refusal is the expected outcome, not a crash.
    from pipekit import Identity, Sequential

    from geoops._src.contract import assert_config_round_trips

    assert_config_round_trips(Sequential([Identity(), Identity()]))


def test_carrier_is_the_shared_toolkit() -> None:
    # The public names are the very objects the built-in operators use.
    from geoops._src import shape, valid, wrap

    assert wrap_like is wrap.wrap_like
    assert mask_invalid_to_nan is valid.mask_invalid_to_nan
    assert over_frames is shape.over_frames


def test_carrier_publishes_the_moving_families_helpers() -> None:
    """The names `plume` and `matched_filter` need are the `_src` objects (#480)."""
    import geoops.carrier as carrier
    from geoops._src import bands, config, geo, optional, samples, shape

    for module in (bands, config, geo, optional, samples, shape):
        for name in carrier.__all__:
            if hasattr(module, name):
                assert getattr(carrier, name) is getattr(module, name), name


@pytest.mark.parametrize("family", ["plume", "matched_filter"])
def test_moving_families_import_no_private_helpers(family: str) -> None:
    """Gate P6: code moving to plumax.geo / radtrax.geo uses public names only.

    Their own ``<family>._src`` is moving with them; every other geotoolz
    name must come from a public module (#480).
    """
    import ast
    from pathlib import Path

    import geoops

    src = Path(geoops.__file__).parent / family / "_src"
    for path in src.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                own = module.startswith(f"geoops.{family}.")
                assert own or "._src" not in module, f"{path.name}: {module}"
