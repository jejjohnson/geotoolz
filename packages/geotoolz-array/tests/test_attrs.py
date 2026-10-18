"""`FrozenAttrs`: freezing rules and hashing (gate GA3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geoarray import FrozenAttrs


def test_lists_become_tuples_recursively() -> None:
    attrs = FrozenAttrs(band_names=["B11", "B12"], nested=[[1, 2], [3]])
    assert attrs["band_names"] == ("B11", "B12")
    assert attrs["nested"] == ((1, 2), (3,))


def test_numeric_arrays_and_numpy_scalars_become_python() -> None:
    wavelengths = np.linspace(381.0, 2493.0, 285)
    attrs = FrozenAttrs(wavelengths=wavelengths, gain=np.float32(2.5), n=np.int64(3))
    assert isinstance(attrs["wavelengths"], tuple)
    assert len(attrs["wavelengths"]) == 285
    assert all(type(v) is float for v in attrs["wavelengths"])
    assert attrs["gain"] == 2.5
    assert type(attrs["gain"]) is float
    assert type(attrs["n"]) is int


def test_mappings_freeze_and_thaw() -> None:
    attrs = FrozenAttrs(tags={"AREA_OR_POINT": "Area", "scales": [1, 2]})
    assert isinstance(attrs["tags"], FrozenAttrs)
    assert attrs.to_dict() == {"tags": {"AREA_OR_POINT": "Area", "scales": (1, 2)}}
    assert FrozenAttrs(attrs) == attrs
    assert FrozenAttrs(tags=attrs["tags"])["tags"] is attrs["tags"]


def test_equal_attrs_hash_equal() -> None:
    a = FrozenAttrs({"band_names": ["B11", "B12"], "sensor": "S2A"})
    b = FrozenAttrs(sensor="S2A", band_names=("B11", "B12"))
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1
    assert a == {"band_names": ("B11", "B12"), "sensor": "S2A"}
    assert a != FrozenAttrs(sensor="S2B", band_names=("B11", "B12"))
    assert (a == 3) is False


def test_nan_values_compare_equal() -> None:
    a = FrozenAttrs(scale=float("nan"), values=[np.nan, 1.0])
    b = FrozenAttrs(scale=np.float64("nan"), values=(float("nan"), 1.0))
    assert a["scale"] is math.nan
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((2, 2)), "only 1-D numeric arrays"),
        (np.array(["a", "b"]), "only 1-D numeric arrays"),
        ({1, 2}, "must be hashable"),
        (bytearray(b"x"), "must be hashable"),
    ],
)
def test_unhashable_values_raise_naming_the_key(value: object, message: str) -> None:
    with pytest.raises(TypeError, match=message) as info:
        FrozenAttrs(bad=value)
    assert "'bad'" in str(info.value)


def test_keys_must_be_strings() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        FrozenAttrs({1: "x"})


def test_hashable_objects_are_kept() -> None:
    marker = frozenset({1, 2})
    assert FrozenAttrs(m=marker)["m"] is marker


def test_set_drop_and_mapping_protocol() -> None:
    attrs = FrozenAttrs(a=1, b=2)
    assert attrs.set(c=3) == {"a": 1, "b": 2, "c": 3}
    assert attrs.set(a=9)["a"] == 9
    assert attrs.drop("b", "missing") == {"a": 1}
    assert len(attrs) == 2
    assert list(attrs) == ["a", "b"]
    assert "a" in attrs
    assert attrs.get("z") is None
    assert repr(FrozenAttrs(a=1)) == "FrozenAttrs({'a': 1})"


def test_frozen_attrs_are_immutable() -> None:
    attrs = FrozenAttrs(a=1)
    with pytest.raises(TypeError):
        attrs["a"] = 2  # type: ignore[index]


def test_pickle_round_trip_in_another_process() -> None:
    """A FrozenAttrs pickled in one process is equal (and hashes equal) in another."""
    import pickle
    import subprocess
    import sys

    attrs = FrozenAttrs(band_names=["B11", "B12"], scale=float("nan"), tags={"a": 1})
    payload = pickle.dumps(attrs).hex()
    code = (
        "import math, pickle, sys\n"
        "from geoarray import FrozenAttrs\n"
        f"a = pickle.loads(bytes.fromhex({payload!r}))\n"
        "b = FrozenAttrs(\n"
        "    band_names=['B11', 'B12'], scale=float('nan'), tags={'a': 1}\n"
        ")\n"
        "assert a == b and hash(a) == hash(b) and a['scale'] is math.nan, a\n"
    )
    env = {"PYTHONHASHSEED": "12345"}
    import os

    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, **env})
