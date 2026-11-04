"""`geoops.carrier.Coupling` and the probes in `geoops.testing` (#615)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _helpers import toy_geoarray
from geoarray import GeoArray
from scipy import ndimage

from geoops.carrier import Coupling, reach_of
from geoops.testing import assert_coupling, measure_coupling, measure_frames


def _scene(bands: int = 3) -> GeoArray:
    values = np.random.default_rng(0).uniform(0.1, 1.0, (bands, 16, 16))
    return toy_geoarray(values, fill_value=None)


class _Toy:
    """A callable operator stand-in with a declared coupling."""

    coupling: Coupling = Coupling()

    def __call__(self, x: Any) -> Any:
        return x.replace(data=self.f(np.asarray(x)))

    def f(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Scale(_Toy):
    coupling = Coupling()

    def f(self, a: np.ndarray) -> np.ndarray:
        return 2.0 * a


class Ratio(_Toy):
    coupling = Coupling(couples={"band"}, effects={"band": "reduce"})

    def __call__(self, x: Any) -> Any:
        a = np.asarray(x)
        return x.replace(data=(a[..., 0:1, :, :] / a[..., 1:2, :, :]))


class Box3(_Toy):
    coupling = Coupling(couples={"space"}, reach=1)

    def f(self, a: np.ndarray) -> np.ndarray:
        size = (1,) * (a.ndim - 2) + (3, 3)
        return ndimage.uniform_filter(a, size=size, mode="nearest")


class MinusSceneMean(_Toy):
    coupling = Coupling(couples={"space"}, reach=None)

    def f(self, a: np.ndarray) -> np.ndarray:
        return a - a.mean(axis=(-2, -1), keepdims=True)


class FitOnCall(_Toy):
    """x - mean_c, the per-band mean fitted over the first scene seen."""

    coupling = Coupling(fit={"space"})

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def f(self, a: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            self.mean_ = a.mean(axis=(-2, -1), keepdims=True)
        return a - self.mean_


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("couples", "name"),
    [
        ((), "pointwise"),
        ({"band"}, "spectral"),
        ({"space"}, "spatial"),
        ({"time"}, "temporal"),
        ({"level", "space"}, "volumetric"),
        ({"band", "space"}, "spatio-spectral"),
        ({"space", "time"}, "spatio-temporal"),
        ({"band", "space", "time"}, "spatio-spectro-temporal"),
    ],
)
def test_class_names(couples: set[str], name: str) -> None:
    reach = None if "space" in couples else 0
    assert Coupling(couples=couples, reach=reach).name == name


def test_coupling_normalises_and_hashes() -> None:
    a = Coupling(couples={"band"}, effects={"band": "reduce", "space": "keep"})
    b = Coupling(couples=frozenset({"band"}), effects=[("band", "reduce")])
    assert a == b
    assert hash(a) == hash(b)
    assert a.effect("band") == "reduce"
    assert a.effect("space") == "keep"
    assert a.effects == (("band", "reduce"),)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"couples": {"pixel"}}, ValueError),
        ({"couples": "band"}, TypeError),
        ({"effects": {"band": "squash"}}, ValueError),
        ({"effects": {"colour": "reduce"}}, ValueError),
        ({"kind": "table"}, ValueError),
        ({"couples": {"space"}, "reach": -1}, ValueError),
        ({"couples": {"space"}, "reach": 0}, ValueError),
        ({"reach": 3}, ValueError),
        ({"couples": {"space"}, "reach": 1.5}, TypeError),
        ({"couples": {"space"}, "reach": None, "fit": {"space"}}, ValueError),
    ],
)
def test_coupling_rejects(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        Coupling(**kwargs)


def test_reach_of_prefers_spatial_reach() -> None:
    class Window:
        coupling = Coupling(couples={"space"}, reach=1)

        def __init__(self, window: int) -> None:
            self.window = window

        def spatial_reach(self) -> int:
            return self.window // 2

    assert reach_of(Window(7)) == 3
    assert reach_of(Box3()) == 1
    with pytest.raises(TypeError, match="declares no Coupling"):
        reach_of(object())


# ---------------------------------------------------------------------------
# measure_coupling
# ---------------------------------------------------------------------------


def test_measures_pointwise() -> None:
    measured = measure_coupling(Scale, _scene())
    assert measured.name == "pointwise"
    assert measured.reach == 0
    assert measured.time == "per frame"
    assert_coupling(Scale(), measured, exact=True)


def test_measures_spectral() -> None:
    measured = measure_coupling(Ratio, _scene())
    assert measured.couples == {"band"}
    assert measured.effects == {"band": "reduce"}
    assert "reads 2 of 3 bands" in measured.notes
    assert_coupling(Ratio(), measured, exact=True)


def test_measures_a_local_reach() -> None:
    measured = measure_coupling(Box3, _scene())
    assert measured.couples == {"space"}
    assert measured.reach == 1
    assert_coupling(Box3(), measured, exact=True)


def test_measures_a_global_operator() -> None:
    measured = measure_coupling(MinusSceneMean, _scene())
    assert measured.reach is None
    assert measured.frozen_space is True
    assert_coupling(MinusSceneMean(), measured, exact=True)


def test_measures_a_fit_over_space() -> None:
    measured = measure_coupling(FitOnCall, _scene())
    assert measured.couples == {"space"}
    assert measured.frozen_space is False
    assert_coupling(FitOnCall(), measured, exact=True)


def test_measures_pooling_over_time() -> None:
    class StackMean(_Toy):
        coupling = Coupling(couples={"time"})

        def f(self, a: np.ndarray) -> np.ndarray:
            return a - a.mean(axis=0, keepdims=True) if a.ndim == 4 else a

    measured = measure_coupling(StackMean, _scene())
    assert measured.time == "pooled"
    assert measured.couples == {"time"}
    assert_coupling(StackMean(), measured, exact=True)


def test_measures_a_regrid() -> None:
    class Every2nd:
        coupling = Coupling(effects={"space": "regrid"})

        def __call__(self, x: GeoArray) -> GeoArray:
            return x[..., ::2, ::2]

    measured = measure_coupling(Every2nd, _scene())
    assert measured.regrids
    assert measured.effects == {"space": "regrid"}
    assert_coupling(Every2nd(), measured, exact=True)


def test_a_band_permutation_is_not_a_coupling() -> None:
    class Reverse(_Toy):
        coupling = Coupling(effects={"band": "select"})

        def f(self, a: np.ndarray) -> np.ndarray:
            return a[..., ::-1, :, :]

    measured = measure_coupling(Reverse, _scene())
    assert measured.couples == frozenset()
    assert measured.effects == {"band": "select"}
    assert_coupling(Reverse(), measured, exact=True)


def test_an_undeclared_regrid_fails() -> None:
    class Every2nd:
        coupling = Coupling()

        def __call__(self, x: GeoArray) -> GeoArray:
            return x[..., ::2, ::2]

    with pytest.raises(AssertionError, match="regrid"):
        assert_coupling(Every2nd(), measure_coupling(Every2nd, _scene()))


def test_stochastic_operators_are_not_judged() -> None:
    class Noise:
        coupling = Coupling()

        def __call__(self, x: GeoArray) -> GeoArray:
            noise = np.random.default_rng().normal(size=x.shape)
            return x.replace(data=np.asarray(x) + noise)

    measured = measure_coupling(Noise, _scene())
    assert measured.stochastic
    assert_coupling(Noise(), measured, exact=True)


def test_rejects_a_small_sample() -> None:
    with pytest.raises(ValueError, match="12 x 12"):
        measure_coupling(Scale, toy_geoarray(np.ones((8, 8))))


# ---------------------------------------------------------------------------
# measure_frames
# ---------------------------------------------------------------------------


def _frames() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.uniform(size=(16, 16)) for _ in range(3)]


def test_measures_a_temporal_reducer() -> None:
    class Median:
        coupling = Coupling(couples={"time"}, effects={"time": "reduce"})

        def __call__(self, *xs: np.ndarray) -> np.ndarray:
            return np.median(np.stack(xs), axis=0)

    measured = measure_frames(Median, _frames())
    assert measured.name == "temporal"
    assert measured.effects == {"time": "reduce"}
    assert_coupling(Median(), measured, exact=True)


def test_measures_a_frame_selector() -> None:
    class First:
        coupling = Coupling(effects={"time": "select"})

        def __call__(self, xs: list[np.ndarray]) -> np.ndarray:
            return xs[0]

    measured = measure_frames(First, _frames(), as_list=True)
    assert measured.name == "pointwise"
    assert measured.effects == {"time": "select"}


def test_measures_a_band_join() -> None:
    measured = measure_frames(lambda: lambda *xs: np.stack(xs), _frames())
    assert measured.effects == {"band": "join"}


# ---------------------------------------------------------------------------
# assert_coupling
# ---------------------------------------------------------------------------


def test_a_too_narrow_declaration_fails() -> None:
    class Undeclared(Box3):
        coupling = Coupling()

    with pytest.raises(AssertionError, match=r"Undeclared coupling.*\['space'\]"):
        assert_coupling(Undeclared(), measure_coupling(Box3, _scene()))


def test_a_too_small_reach_fails() -> None:
    class Box5(_Toy):
        coupling = Coupling(couples={"space"}, reach=1)

        def f(self, a: np.ndarray) -> np.ndarray:
            size = (1,) * (a.ndim - 2) + (5, 5)
            return ndimage.uniform_filter(a, size=size, mode="nearest")

    with pytest.raises(AssertionError, match="measured reach 2 px > declared 1 px"):
        assert_coupling(Box5(), measure_coupling(Box5, _scene()))


def test_a_local_declaration_of_a_global_operator_fails() -> None:
    class LocalClaim(MinusSceneMean):
        coupling = Coupling(couples={"space"}, reach=2)

    with pytest.raises(AssertionError, match="declare reach=None"):
        assert_coupling(LocalClaim(), measure_coupling(MinusSceneMean, _scene()))


def test_a_wider_declaration_passes_unless_exact() -> None:
    class Wide(Scale):
        coupling = Coupling(couples={"band", "space"}, reach=None)

    measured = measure_coupling(Scale, _scene())
    assert_coupling(Wide(), measured)
    with pytest.raises(AssertionError, match="exact"):
        assert_coupling(Wide(), measured, exact=True)


def test_an_undeclared_operator_fails() -> None:
    with pytest.raises(AssertionError, match="declares no `coupling`"):
        assert_coupling(object(), measure_coupling(Scale, _scene()))


def test_reads_values_is_checked() -> None:
    class GridOnly:
        coupling = Coupling(reads_values=False)

        def __call__(self, x: GeoArray) -> GeoArray:
            return x.replace(data=np.zeros(x.shape, dtype=bool))

    class ClaimsGridOnly(Scale):
        coupling = Coupling(reads_values=False)

    measured = measure_coupling(GridOnly, _scene())
    assert not measured.reads_values
    assert_coupling(GridOnly(), measured, exact=True)
    with pytest.raises(AssertionError, match="reads_values=True"):
        assert_coupling(ClaimsGridOnly(), measure_coupling(Scale, _scene()))
    with pytest.raises(AssertionError, match="reads_values=False"):
        assert_coupling(Scale(), measured, exact=True)


def test_a_reduction_over_time_is_measured() -> None:
    class StackMean:
        coupling = Coupling(couples={"time"}, effects={"time": "reduce"})

        def __call__(self, x: GeoArray) -> np.ndarray:
            a = np.asarray(x)
            return a.mean(axis=0) if a.ndim == 4 else a

    measured = measure_coupling(StackMean, _scene())
    assert measured.time == "reduce"
    assert_coupling(StackMean(), measured, exact=True)
