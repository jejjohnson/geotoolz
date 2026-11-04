"""Declared operator coupling holds against a perturbation probe (#616).

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

from geoops.carrier import Coupling
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
