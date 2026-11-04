"""Declared operator coupling holds against a perturbation probe (#616, #617).

Every operator class that declares ``coupling`` is probed on the first toy
input it accepts (``_coupling_inputs.probe``) and its declaration is checked
with `geoops.testing.assert_coupling`: the declaration must contain what
the probe measured. In the namespaces where the probe reads the maths
exactly, the two must be equal.
"""

from __future__ import annotations

import pytest
from _coupling_inputs import build_for_probe, probe
from _helpers import all_operator_classes
from test_operator_contract import _key

from geoops.carrier import Coupling, reach_of
from geoops.testing import assert_coupling


#: Namespaces whose declarations must equal the measurement.
EXACT_NAMESPACES: tuple[str, ...] = ("indices", "spectral", "radiometry")

#: Operators in those namespaces where the toy input under-reads the maths.
NOT_EXACT: dict[str, str] = {
    "indices._src.operators.dNBR": "two carriers: not probed",
    "spectral._src.operators.SpectralBinning": (
        "toy bins hold one source band each, so no band mixes"
    ),
}


def test_every_operator_declares_coupling() -> None:
    """Every public operator class says which dims it mixes (gate OC-G2)."""
    missing = [
        _key(cls)
        for cls in all_operator_classes()
        if not cls.__name__.startswith("_")
        and not isinstance(getattr(cls, "coupling", None), Coupling)
    ]
    assert missing == [], "declare `coupling` (geoops.carrier.Coupling)"


def _declared() -> list[type]:
    return [
        cls
        for cls in all_operator_classes()
        if not cls.__name__.startswith("_")
        and isinstance(getattr(cls, "coupling", None), Coupling)
    ]


def _exact(key: str) -> bool:
    return key.split(".")[0] in EXACT_NAMESPACES and key not in NOT_EXACT


@pytest.mark.parametrize("cls", _declared(), ids=_key)
def test_declared_coupling_holds(cls: type, tmp_path, monkeypatch) -> None:
    """The declared coupling contains (or, where exact, equals) the measured one."""
    monkeypatch.chdir(tmp_path)  # sinks write their relative paths here
    measured, why = probe(cls)
    if measured is None:
        pytest.skip(why)
    assert_coupling(build_for_probe(cls), measured, exact=_exact(_key(cls)))


def test_a_too_narrow_declaration_fails() -> None:
    """A subclass declaring NDVI pointwise is caught."""
    from geoops.indices import NDVI

    class PointwiseNDVI(NDVI):
        coupling = Coupling()

    measured, _ = probe(NDVI)
    assert measured is not None
    with pytest.raises(AssertionError, match=r"PointwiseNDVI coupling.*\['band'\]"):
        assert_coupling(PointwiseNDVI(), measured)


@pytest.mark.parametrize("cls", _declared(), ids=_key)
def test_reach_of_every_operator(cls: type) -> None:
    """``reach_of`` works on every buildable operator and agrees with ``coupling``."""
    try:
        op = build_for_probe(cls)
    except pytest.skip.Exception:
        pytest.skip("needs runtime constructor objects")
    reach = reach_of(op)
    assert reach is None or (isinstance(reach, int) and reach >= 0)
    assert (reach != 0) == ("space" in op.coupling.couples)


def _wide_scene():
    import numpy as np
    from _helpers import toy_geoarray

    values = np.random.default_rng(0).uniform(0.1, 1.0, (2, 32, 32))
    return toy_geoarray(values, fill_value=None)


@pytest.mark.parametrize(
    "build",
    [
        lambda: __import__("geoops").restore.GaussianDenoise(sigma=2.0),
        lambda: __import__("geoops").restore.MedianDenoise(window=5),
        lambda: __import__("geoops").restore.DespeckleLee(window=9),
        lambda: __import__("geoops").geom.Laplacian2D(accuracy=2),
    ],
    ids=["GaussianDenoise", "MedianDenoise", "DespeckleLee", "Laplacian2D"],
)
def test_spatial_reach_follows_the_arguments(build) -> None:
    """An argument-dependent reach bounds the reach measured on a 32 x 32 scene."""
    from geoops.testing import measure_coupling

    measured = measure_coupling(build, _wide_scene())
    assert measured.reach is not None
    assert 0 < measured.reach <= reach_of(build())
    assert_coupling(build(), measured)


def test_spatial_reach_values() -> None:
    import geoops

    assert reach_of(geoops.restore.GaussianDenoise(sigma=2.0)) == 8
    assert reach_of(geoops.restore.MedianDenoise(window=5)) == 2
    assert reach_of(geoops.mask.DilateMask(iterations=3)) == 3
    assert reach_of(geoops.mask.OpenMask(iterations=2)) == 4
    assert reach_of(geoops.mask.BufferMask(radius=2.5)) == 3
    assert reach_of(geoops.mask.BufferMask(radius=20.0, unit="m")) is None
    assert reach_of(geoops.geom.Gradient2D()) == 1
    assert reach_of(geoops.geom.Laplacian2D(accuracy=2)) == 3
    assert reach_of(geoops.restore.GapFillNearest()) is None
    assert reach_of(geoops.restore.GapFillNearest(max_distance=4)) == 4
