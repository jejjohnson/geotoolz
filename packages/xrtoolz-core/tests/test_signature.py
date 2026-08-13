"""Unit tests for :class:`xrcore.Signature` — the dict-keyed shape descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from xrcore import Signature


def test_equality_is_order_insensitive_and_dtype_normalized() -> None:
    a = Signature({"time": 12, "lat": 4}, dtype="float32")
    b = Signature({"lat": 4, "time": 12}, dtype=np.float32)

    assert a == b
    assert hash(a) == hash(b)


def test_dims_are_frozen() -> None:
    sig = Signature({"time": 12})

    with pytest.raises(TypeError):
        sig.dims["time"] = 5  # type: ignore[index]


def test_unknown_size_and_dim_edits() -> None:
    sig = Signature({"time": None, "lat": 4}, dtype="float64")

    renamed = sig.rename_dims({"lat": "y"})
    assert "y" in renamed.dims and "lat" not in renamed.dims

    dropped = sig.drop_dims("time")
    assert set(dropped.dims) == {"lat"}

    replaced = sig.replace_dims({"time": 6})
    assert replaced.dims["time"] == 6


def test_invalid_dims_raise() -> None:
    with pytest.raises(TypeError):
        Signature({1: 2})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        Signature({"time": "twelve"})  # type: ignore[dict-item]
