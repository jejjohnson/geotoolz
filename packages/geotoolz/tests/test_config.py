"""Tests for the shared config coercion (``geotoolz._src.config.jsonable``)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pytest

import geotoolz as gz
from geotoolz._src.config import jsonable


def test_jsonable_bytes_decode_everywhere() -> None:
    value = {
        "title": b"scene",
        "names": [b"red", np.bytes_(b"nir")],
        "array": np.array([b"a", b"b"]),
        "bad": b"\xff",
    }
    assert jsonable(value) == {
        "title": "scene",
        "names": ["red", "nir"],
        "array": ["a", "b"],
        "bad": "�",
    }


def test_jsonable_builtin_coercions() -> None:
    value = {
        1: (np.float32(0.5), np.int64(3), np.bool_(True)),
        "arr": np.arange(4).reshape(2, 2),
        "when": datetime(2024, 1, 2, 3, 4, 5),
        "day": date(2024, 1, 2),
        "dt64": np.datetime64("2024-01-02T03:04:05"),
        "path": PurePosixPath("/tmp/x.tif"),
    }
    assert jsonable(value) == {
        "1": [0.5, 3, True],
        "arr": [[0, 1], [2, 3]],
        "when": "2024-01-02T03:04:05",
        "day": "2024-01-02",
        "dt64": "2024-01-02T03:04:05",
        "path": "/tmp/x.tif",
    }


def test_jsonable_non_strict_passes_unknown_through() -> None:
    obj = object()
    out = jsonable({"obj": obj, "nan": float("nan")})
    assert out["obj"] is obj
    assert np.isnan(out["nan"])


def test_jsonable_strict_reprs_instead_of_dropping() -> None:
    class Thing:
        def __repr__(self) -> str:
            return "Thing()"

    out = jsonable(
        {
            "obj": Thing(),
            "nested": [("step", Thing())],
            "nan": np.float64("nan"),
            "inf": float("inf"),
            "ok": [1, 2.5, "s", None, True],
            "fn": len,
        },
        strict=True,
    )
    assert out == {
        "obj": "Thing()",
        "nested": [["step", "Thing()"]],
        "nan": "nan",
        "inf": "inf",
        "ok": [1, 2.5, "s", None, True],
        "fn": repr(len),
    }
    json.dumps(out, allow_nan=False)


def _old_stat_as_jsonable(value: Any) -> Any:
    """The removed ``normalize._stat_as_jsonable``, pinned for equivalence."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return jsonable(arr)


def _old_coef_as_jsonable(coef: Any) -> Any:
    """The removed ``radiometry._coef_as_jsonable``, pinned for equivalence."""
    arr = np.asarray(coef, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return jsonable(arr.ravel())


@pytest.mark.parametrize(
    "value",
    [
        2,
        np.float32(0.25),
        [1, 2, 3],
        (0.5, 1.5),
        np.array([1, 2], dtype=np.uint16),
        np.array([0.1, 0.2], dtype=np.float32),
    ],
)
def test_float_coercion_matches_removed_helpers(value: Any) -> None:
    new = jsonable(np.asarray(value, dtype=float))
    assert new == _old_stat_as_jsonable(value) == _old_coef_as_jsonable(value)
    assert type(new) is type(_old_coef_as_jsonable(value))


def test_normalize_configs_unchanged() -> None:
    op = gz.normalize.StandardScaler(mean=[1, 2], std=np.array([0.5, 1.5], "f4"))
    assert op.get_config()["mean"] == _old_stat_as_jsonable(op.mean)
    assert op.get_config()["std"] == _old_stat_as_jsonable(op.std)
    assert gz.normalize.StandardScaler().get_config()["mean"] is None

    stats = gz.normalize.PerBandStats(percentiles=[5.0, 95.0])
    stats(np.arange(18, dtype=np.float32).reshape(2, 3, 3))
    for value in stats.stats.values():
        json.dumps(value, allow_nan=False)
    assert np.asarray(stats.stats["percentiles"]).shape == (2, 2)


def test_radiometry_configs_unchanged() -> None:
    op = gz.radiometry.DNToRadiance(gain=[1, 2], offset=np.float32(0.5), scale=3)
    config = op.get_config()
    assert config["gain"] == _old_coef_as_jsonable([1, 2]) == [1.0, 2.0]
    assert config["offset"] == 0.5
    assert config["scale"] == 3.0
    assert type(config["scale"]) is float

    toa = gz.radiometry.RadianceToReflectance(
        solar_irradiance=[1800.0, 1500.0],
        acquisition_date="2024-06-01T10:30:00",
        sza_deg=30.0,
    )
    assert toa.get_config()["acquisition_date"] == "2024-06-01T10:30:00"


def test_learn_params_keep_nested_estimators_and_random_state() -> None:
    pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("km", KMeans(n_clusters=2, random_state=np.random.RandomState(0))),
        ]
    )
    params = gz.learn.SklearnOp(pipe).get_config()["estimator"]["params"]
    assert params["steps"][0] == ["scale", "StandardScaler()"]
    assert params["steps"][1][0] == "km"
    assert params["steps"][1][1].startswith("KMeans(")
    json.dumps(params, allow_nan=False)

    km = gz.learn.KMeans(KMeans(n_clusters=2, random_state=np.random.RandomState(0)))
    params = km.get_config()["estimator"]["params"]
    assert params["random_state"].startswith("RandomState(")
    assert params["n_clusters"] == 2
