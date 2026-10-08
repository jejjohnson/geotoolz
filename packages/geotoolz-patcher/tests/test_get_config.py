"""Config round-trip contract for every axis, stencil and patcher.

For every object whose ``forbid_in_yaml`` is ``False``:

- ``axis_envelope(obj)`` survives a JSON round-trip;
- ``from_config`` rebuilds an object of the same type with the same config
  and the same behaviour (anchors / windows / weights / merges);
- a leaf (no nested envelopes) also rebuilds as ``type(obj)(**cfg)``.

Objects flagged ``forbid_in_yaml`` must be refused by `from_config`.
Every patcher config nests each component via `axis_envelope`.

`test_every_component_class_has_an_example` keeps the example table
complete: a new concrete axis without an entry here fails the suite.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rasterio
from _helpers import make_raster_field
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

import geopatcher
from geopatcher import (
    AsyncSpatialPatcher,
    Patch,
    SpatialPatcher,
    SpatioTemporalPatcher,
    TemporalPatch,
    TemporalPatcher,
    spatial,
    temporal,
)
from geopatcher._src.spatial.patcher import _matches_retry_on
from geopatcher.config import axis_envelope, from_config
from geopatcher.fields import PointDomain
from geopatcher.matched import (
    MatchedSpatialPatcher,
    MatchedSpatioTemporalPatcher,
    MatchedTemporalPatcher,
)
from geopatcher.run import PatchCache
from geopatcher.temporal.stencils import Stencil, TimeStencil


def _hourly(n: int = 24) -> np.ndarray:
    return np.datetime64("2024-01-01T00", "h") + np.arange(n) * np.timedelta64(1, "h")


def _stencil() -> TimeStencil:
    return TimeStencil("-3h", "0h", "1h", closed="both")


def _spatial_patcher(**kwargs: Any) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8), boundary="pad"),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(chunks=(4, 4)),
        **kwargs,
    )


def _temporal_patcher() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=4),
        sampler=temporal.sampler.RegularStride(step=2),
        window=temporal.window.ExponentialDecay(tau=2.0),
        aggregation=temporal.aggregation.Mean(),
    )


def _polygons() -> Any:
    import geopandas as gpd
    import shapely

    return gpd.GeoSeries([shapely.box(0, 0, 4, 4)])


EXAMPLES: list[Any] = [
    # -- spatial geometries
    spatial.geometry.Rectangular(size=(8, 8)),
    spatial.geometry.Rectangular(size=(4, 6), boundary="pad", pad_value=-1.0),
    spatial.geometry.SphericalCap(radius_km=500.0),
    spatial.geometry.KNNGraph(k=4, metric="haversine"),
    spatial.geometry.RadiusGraph(radius=10.0),
    spatial.geometry.PolygonIntersection(polygons=_polygons()),
    # -- spatial samplers
    spatial.sampler.RegularStride(step=(4, 6)),
    spatial.sampler.JitteredStride(step=8, jitter=0.5, seed=0),
    spatial.sampler.Random(n_samples=4, seed=0),
    spatial.sampler.PoissonDisk(min_dist=4.0, seed=0),
    spatial.sampler.Explicit(anchors_=[(0, 0), (8, 8)]),
    spatial.sampler.AlongTrack(
        track=[(1.5, 2.5), (20.25, 9.0), (30.0, 30.0)], spacing=3.0
    ),
    spatial.sampler.ExplicitCoords(coords=[(3.0, 4.0), (17.5, 2.0)], crs="EPSG:32630"),
    # -- spatial windows
    spatial.window.Boxcar(),
    spatial.window.Hann(),
    spatial.window.Tukey(alpha=0.4),
    spatial.window.Gaussian(sigma=0.4),
    spatial.window.Custom(fn=lambda g: np.ones(g.size)),
    # -- spatial aggregations
    spatial.aggregation.Sum(),
    spatial.aggregation.Max(fill_value=-1.0),
    spatial.aggregation.Min(),
    spatial.aggregation.WeightedSum(),
    spatial.aggregation.WeightedSum(weight_fn=lambda p: np.ones(np.shape(p.data))),
    spatial.aggregation.Mean(),
    spatial.aggregation.Variance(),
    spatial.aggregation.MeanStd(),
    spatial.aggregation.MinMax(),
    spatial.aggregation.OverlapAdd(),
    spatial.aggregation.OverlapAdd(
        chunks=(4, 4), cog={"blocksize": 256}, dtype="float64"
    ),
    spatial.aggregation.InvVarWeightedMean(),
    spatial.aggregation.HardVote(n_classes=3),
    spatial.aggregation.SoftVote(n_classes=3),
    spatial.aggregation.ByIndex(),
    spatial.aggregation.Median(),
    spatial.aggregation.Mode(),
    spatial.aggregation.Learned(model=lambda chips: chips[0]),
    spatial.aggregation.ApproxQuantile(),
    spatial.aggregation.Reservoir(k=10, seed=0),
    spatial.aggregation.ApproxCardinality(),
    spatial.aggregation.ApproxMode(),
    spatial.aggregation.StreamingHistogram(),
    # -- stencils
    Stencil(start=-2, stop=2, step=0.5, closed="both"),
    Stencil(
        start=np.timedelta64(-3, "h"),
        stop=np.timedelta64(0, "h"),
        step=np.timedelta64(1, "h"),
    ),
    TimeStencil("-9h", "3h", "1h", closed="both"),
    TimeStencil(
        np.timedelta64(-3000, "ms"), np.timedelta64(0, "ms"), np.timedelta64(1000, "ms")
    ),
    TimeStencil("-2W", "0W", "1W", closed="both"),
    # -- temporal geometries
    temporal.geometry.FixedLookback(length=5),
    temporal.geometry.LookbackHorizon(lookback=4, horizon=2),
    temporal.geometry.MultiScale(scales=[2, 4, 8]),
    temporal.geometry.StencilGeometry(
        stencil=_stencil(), source_step=np.timedelta64(1, "h")
    ),
    temporal.geometry.StencilGeometry(stencil=_stencil()),
    temporal.geometry.StencilGeometry(stencil=_stencil(), boundary="shrink"),
    temporal.geometry.PhaseWindow(period=24, phase_width=2),
    # -- temporal samplers
    temporal.sampler.RegularStride(step=2),
    temporal.sampler.RegularStride(step=1, start=3),
    temporal.sampler.Explicit(times=[3, 5, 7]),
    temporal.sampler.Random(n_samples=4, seed=0),
    temporal.sampler.StencilSampler(stencil=_stencil(), every=2, shuffle=True, seed=0),
    temporal.sampler.Explicit(times=[0, 4, 8]),
    # -- temporal windows
    temporal.window.CausalBoxcar(),
    temporal.window.ExponentialDecay(tau=2.0),
    temporal.window.TaperedTukey(alpha=0.4),
    temporal.window.Periodic(period=24),
    # -- temporal aggregations
    temporal.aggregation.Fold(fold_fn=lambda acc, x: x),
    temporal.aggregation.Mean(),
    temporal.aggregation.HierarchicalCombine(scales=[2, 4]),
    temporal.aggregation.Forecast(horizon=2),
    # -- patchers
    _spatial_patcher(),
    _spatial_patcher(
        on_error="retry",
        max_retries=2,
        retry_on=(RasterioIOError, "TimeoutError"),
        capture_traceback=False,
    ),
    AsyncSpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.Mean(),
        on_error="skip",
    ),
    _temporal_patcher(),
    TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=4),
        sampler=temporal.sampler.RegularStride(step=2, start=3, check_full_scan=False),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
        on_error="retry",
        max_retries=1,
        retry_on=(OSError, "TimeoutError"),
        capture_traceback=False,
    ),
    TemporalPatcher(
        geometry=temporal.geometry.StencilGeometry(stencil=_stencil()),
        sampler=temporal.sampler.StencilSampler(stencil=_stencil()),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    ),
    SpatioTemporalPatcher(
        spatial=_spatial_patcher(),
        temporal=_temporal_patcher(),
        coupling="coupled",
        time_axis=1,
    ),
    MatchedSpatialPatcher(
        primary=_spatial_patcher(),
        secondary_aggregators={"s2": spatial.aggregation.Mean()},
    ),
    MatchedTemporalPatcher(
        primary=_temporal_patcher(),
        secondary_aggregators={"era5": temporal.aggregation.Mean()},
    ),
    MatchedSpatioTemporalPatcher(
        primary=SpatioTemporalPatcher(
            spatial=_spatial_patcher(), temporal=_temporal_patcher()
        ),
        secondary_aggregators={"era5": temporal.aggregation.Mean()},
    ),
]

PATCHER_TYPES = (
    SpatialPatcher,
    AsyncSpatialPatcher,
    TemporalPatcher,
    SpatioTemporalPatcher,
    MatchedSpatialPatcher,
    MatchedTemporalPatcher,
    MatchedSpatioTemporalPatcher,
)


def _id(obj: Any) -> str:
    return type(obj).__name__


def _dumps(cfg: Any) -> str:
    # String compare: NaN fill values never compare equal as floats.
    return json.dumps(cfg, sort_keys=True)


def _has_envelope(value: Any) -> bool:
    if isinstance(value, dict):
        if set(value) == {"class", "config"}:
            return True
        return any(_has_envelope(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_envelope(v) for v in value)
    return False


# -- behaviour probes --------------------------------------------------------

_RASTER = SimpleNamespace(
    transform=rasterio.Affine.identity(), shape=(32, 32), crs="EPSG:32630"
)


def _point_domain() -> PointDomain:
    from scipy.spatial import cKDTree

    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [5.0, 5.0], [9.0, 1.0]])
    return PointDomain(coords=coords, kdtree=cKDTree(coords))


def _spatial_patches(agg: spatial.aggregation.Aggregation) -> list[Patch]:
    rng = np.random.default_rng(0)

    def data() -> Any:
        if isinstance(agg, spatial.aggregation.InvVarWeightedMean):
            return rng.normal(size=(8, 8)), rng.uniform(0.5, 2.0, size=(8, 8))
        if isinstance(agg, spatial.aggregation.SoftVote):
            return rng.dirichlet(np.ones(3), size=(8, 8)).transpose(2, 0, 1)
        return rng.integers(0, 3, size=(8, 8)).astype(float)

    return [
        Patch(data=data(), anchor=(r, c), indices=Window(c, r, 8, 8))
        for r, c in [(0, 0), (4, 4), (8, 0)]
    ]


def _behaviour(obj: Any) -> Any:
    """What the object *does* — compared between original and rebuild."""
    if isinstance(obj, spatial.geometry.Geometry):
        if isinstance(obj, spatial.geometry.Rectangular):
            return obj.neighborhood(_RASTER, (2, 3))
        return obj.neighborhood(_point_domain(), (1.0, 1.0))
    if isinstance(obj, spatial.sampler.Sampler):
        return list(obj.anchors(_RASTER, spatial.geometry.Rectangular(size=(4, 4))))
    if isinstance(obj, spatial.window.Window):
        return obj.weights(spatial.geometry.Rectangular(size=(5, 6)))
    if isinstance(obj, spatial.aggregation.Aggregation):
        return obj.merge(_spatial_patches(obj), SimpleNamespace(shape=(16, 16)))
    if isinstance(obj, Stencil):
        return obj.points
    if isinstance(obj, temporal.geometry.StencilGeometry):
        return obj.window_coord(_hourly(), 10)
    if isinstance(obj, temporal.geometry.Geometry):
        return obj.window(24, 10)
    if isinstance(obj, temporal.sampler.Sampler):
        if obj.needs_coord:
            return list(obj.anchors(24, coord=_hourly()))
        return list(obj.anchors(24))
    if isinstance(obj, temporal.window.Window):
        return obj.weights(temporal.geometry.FixedLookback(length=6), 6)
    if isinstance(obj, temporal.aggregation.Aggregation):
        patches = [
            TemporalPatch(
                data=np.full(4, float(t)), anchor=t, indices=slice(t - 3, t + 1)
            )
            for t in (3, 5, 7)
        ]
        return obj.merge(patches)
    if isinstance(obj, SpatialPatcher):
        return [p.anchor for p in obj.split(make_raster_field(32))]
    if isinstance(obj, TemporalPatcher):
        coord = _hourly() if obj.geometry.needs_coord else None
        return [(p.anchor, p.indices) for p in obj.split(np.arange(24.0), coord=coord)]
    return None


def _outcome(obj: Any) -> tuple[str, Any]:
    try:
        return "ok", _behaviour(obj)
    except Exception as exc:  # the rebuild must fail the same way
        return "error", type(exc)


def _assert_same(a: Any, b: Any) -> None:
    if isinstance(a, Window):
        assert a == b
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    else:
        np.testing.assert_equal(a, b)


# -- the contract -------------------------------------------------------------


def test_every_component_class_has_an_example() -> None:
    bases = (
        spatial.geometry.Geometry,
        spatial.sampler.Sampler,
        spatial.window.Window,
        spatial.aggregation.Aggregation,
        temporal.geometry.Geometry,
        temporal.sampler.Sampler,
        temporal.window.Window,
        temporal.aggregation.Aggregation,
    )
    public = {
        obj
        for name in geopatcher.__all__
        if isinstance(obj := getattr(geopatcher, name), type)
        and issubclass(obj, bases)
        and obj not in bases
    }
    covered = {type(obj) for obj in EXAMPLES}
    assert public - covered == set()
    assert set(PATCHER_TYPES) <= covered
    assert {Stencil, TimeStencil} <= covered


@pytest.mark.parametrize("obj", EXAMPLES, ids=_id)
def test_rebuild_from_config(obj: Any) -> None:
    envelope = json.loads(json.dumps(axis_envelope(obj)))
    if getattr(obj, "forbid_in_yaml", False):
        with pytest.raises((RuntimeError, TypeError)):
            from_config(envelope)
        return

    rebuilt = from_config(envelope)

    assert type(rebuilt) is type(obj)
    assert _dumps(rebuilt.get_config()) == _dumps(obj.get_config())
    expected, actual = _outcome(obj), _outcome(rebuilt)
    assert expected[0] == actual[0]
    _assert_same(expected[1], actual[1])
    if not _has_envelope(envelope["config"]):
        leaf = type(obj)(**envelope["config"])
        assert _dumps(leaf.get_config()) == _dumps(obj.get_config())


@pytest.mark.parametrize(
    "obj", [o for o in EXAMPLES if isinstance(o, PATCHER_TYPES)], ids=_id
)
def test_patcher_configs_nest_via_axis_envelope(obj: Any) -> None:
    cfg = obj.get_config()
    for name, value in vars(obj).items():
        if name in cfg and hasattr(value, "get_config"):
            assert cfg[name] == axis_envelope(value)
        if name in cfg and isinstance(value, dict):
            assert cfg[name] == {k: axis_envelope(v) for k, v in value.items()}


def test_stencil_axes_nest_the_stencil_class() -> None:
    cfg = temporal.geometry.StencilGeometry(stencil=_stencil()).get_config()
    assert cfg["stencil"]["class"] == "temporal.stencils.TimeStencil"
    # A TimeStencil config fed to the generic Stencil still builds timedeltas.
    plain = Stencil(**cfg["stencil"]["config"])
    np.testing.assert_equal(plain.points, _stencil().points)


@pytest.mark.parametrize("unit", ["ms", "us", "ns", "s", "m", "h", "D", "W"])
def test_time_stencil_reloads_every_timedelta_unit(unit: str) -> None:
    td = np.timedelta64
    stencil = TimeStencil(td(-4, unit), td(0, unit), td(1, unit), closed="both")
    # Both the config form and NumPy's own str() form ("-4 milliseconds").
    assert TimeStencil(**stencil.get_config()) == stencil
    assert (
        TimeStencil(
            str(td(-4, unit)), str(td(0, unit)), str(td(1, unit)), closed="both"
        )
        == stencil
    )


def test_source_step_reloads() -> None:
    geom = temporal.geometry.StencilGeometry(
        stencil=_stencil(), source_step=np.timedelta64(1, "h")
    )
    assert geom.get_config()["source_step"] == {"value": 1, "unit": "h"}
    rebuilt = from_config(json.loads(json.dumps(axis_envelope(geom))))
    assert rebuilt.source_step == np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="stride-1"):
        temporal.geometry.StencilGeometry(stencil=_stencil(), source_step="30m")


class TestRetryOn:
    def test_roundtrip_keeps_subclass_semantics(self) -> None:
        patcher = _spatial_patcher(on_error="retry", max_retries=1, retry_on=(OSError,))
        rebuilt = from_config(json.loads(json.dumps(axis_envelope(patcher))))
        exc = RasterioIOError("transient")  # an OSError subclass
        assert _matches_retry_on(exc, patcher.retry_on)
        assert _matches_retry_on(exc, rebuilt.retry_on)

    def test_qualified_name_matches(self) -> None:
        exc = RasterioIOError("transient")
        assert _matches_retry_on(exc, ("rasterio.errors.RasterioIOError",))
        assert _matches_retry_on(exc, ("RasterioIOError",))
        assert not _matches_retry_on(ValueError("bug"), ("OSError",))

    def test_serialised_as_qualified_names(self) -> None:
        patcher = _spatial_patcher(retry_on=(RasterioIOError, TimeoutError, "X"))
        assert patcher.get_config()["retry_on"] == [
            "rasterio.errors.RasterioIOError",
            "TimeoutError",
            "X",
        ]

    def test_rejects_non_exception_entries(self) -> None:
        with pytest.raises(TypeError, match="retry_on"):
            _spatial_patcher(retry_on=(42,))


class TestSummaryConfigs:
    """The axes whose configs used to be ``{"n_...": len(...)}`` summaries."""

    def test_array_backed_axes_dump_their_values(self) -> None:
        assert temporal.sampler.Explicit(times=[0, 4]).get_config() == {"times": [0, 4]}
        assert temporal.sampler.Explicit(times=[3]).get_config() == {"times": [3]}
        cfg = spatial.sampler.ExplicitCoords(coords=[(3.0, 4.0)]).get_config()
        assert cfg["coords"] == [[3.0, 4.0]]

    def test_runtime_object_axes_are_flagged(self) -> None:
        assert spatial.sampler.Explicit.forbid_in_yaml is True
        assert spatial.geometry.PolygonIntersection.forbid_in_yaml is True
        assert spatial.aggregation.WeightedSum().forbid_in_yaml is False
        assert spatial.aggregation.WeightedSum(
            weight_fn=lambda p: p.weights
        ).forbid_in_yaml

    def test_polygon_geometry_is_not_a_cache_key(self) -> None:
        geom = spatial.geometry.PolygonIntersection(polygons=_polygons())
        with pytest.raises(ValueError, match="forbid_in_yaml"):
            PatchCache.config_id_for(geom, spatial.window.Boxcar())

    def test_weight_fn_config_does_not_rebuild(self) -> None:
        agg = spatial.aggregation.WeightedSum(weight_fn=math.sqrt)
        assert agg.get_config()["weight_fn"] == "sqrt"
        with pytest.raises(TypeError, match="weight_fn"):
            from_config(axis_envelope(agg))


def test_async_spatial_patcher_get_config() -> None:
    patcher = AsyncSpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.Mean(),
    )
    sync = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.Mean(),
    )
    assert patcher.get_config() == sync.get_config()


def test_from_config_rejects_unknown_and_malformed() -> None:
    with pytest.raises(TypeError, match="not a loaded"):
        from_config({"class": "NoSuchAxis", "config": {}})
    with pytest.raises(TypeError, match="envelope"):
        from_config({"size": [8, 8]})
