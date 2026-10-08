"""Package-wide ``pipekit.Operator`` contract tests.

Walks every ``geotoolz`` operator class and checks, without optional
extras such as hydra-zen, that:

* ``Operator.from_state(op.state)`` rebuilds an equal operator, or raises
  ``RuntimeError`` for ``forbid_in_yaml`` classes;
* ``get_config()`` is strict JSON (no NaN / inf) and names only
  constructor parameters;
* calling the operator on ``Input`` nodes builds a graph ``Node``, and
  every multi-input operator (#141) runs inside a ``pipekit.Graph`` as
  ``op(Input("a"), Input("b"))`` with the eager result, and rejects a
  second carrier on another pixel grid with an error naming itself.
* on a 4-D ``(T, C, H, W)`` time stack the operator returns the per-frame
  results restacked along time, or rejects the rank with a clear error
  naming itself (``test_time_stack_contract``).

Operators that need constructor arguments get them from ``CTOR_KWARGS``;
runtime objects (geometries, callables) for the ``forbid_in_yaml`` classes
come from ``RUNTIME_CTOR_KWARGS`` in the checks that never serialise the
operator. Second carriers are call arguments, never constructor kwargs
(#141): ``MULTI_INPUTS`` builds them on the primary input's grid.
Known contract violations are listed in ``KNOWN_FAILURES`` as strict
xfails pointing at the issue that fixes them; fixing one makes the
xfail pass, so remove the entry in the same PR.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _helpers import all_operator_classes, fill_pixel_mask
from pipekit import Input, Node, Operator

from geotoolz._src.contract import (
    assert_clear_rank_error,
    assert_fill_matches_dtype,
    assert_fresh_consistent_attrs,
    geotensors,
)


def _key(cls: type) -> str:
    return f"{cls.__module__.removeprefix('geotoolz.')}.{cls.__qualname__}"


def _sklearn(name: str) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        import sklearn.cluster
        import sklearn.covariance
        import sklearn.decomposition
        import sklearn.ensemble
        import sklearn.experimental.enable_iterative_imputer
        import sklearn.impute
        import sklearn.mixture
        import sklearn.neighbors
        import sklearn.svm

        for mod in (
            sklearn.cluster,
            sklearn.decomposition,
            sklearn.ensemble,
            sklearn.impute,
            sklearn.mixture,
            sklearn.neighbors,
            sklearn.svm,
        ):
            if hasattr(mod, name):
                return {"estimator": getattr(mod, name)()}
        raise LookupError(name)

    return build


def _mf(**extra: Any) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        return {"cov_op": np.eye(3), "target": np.ones(3), **extra}

    return build


_DATE = "2024-06-01T10:30:00"
_BOUNDS = (500_000.0, 3_999_840.0, 500_160.0, 4_000_000.0)


def _natural_earth_stub() -> str:
    """A tiny local vector file, so Natural Earth masks never download."""
    import tempfile

    import geopandas as gpd
    from shapely.geometry import box

    path = f"{tempfile.gettempdir()}/geotoolz-contract-natural-earth.gpkg"
    gpd.GeoDataFrame(
        {"ISO_A3": ["GRL"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    ).to_file(path)
    return path


#: Constructor kwargs for operators that cannot be built with no arguments.
#: Values are dicts, or zero-arg callables returning one (for runtime objects).
#: ``forbid_in_yaml`` classes need no entry: their round-trip is checked on
#: a synthetic state record.
CTOR_KWARGS: dict[str, dict[str, Any] | Callable[[], dict[str, Any]]] = {
    "augment._src.operators.Compose": lambda: {
        "augmentations": [
            __import__("geotoolz.augment", fromlist=["RandomFlip"]).RandomFlip()
        ]
    },
    "augment._src.operators.RandomCrop": {"size": (4, 4)},
    "augment._src.operators.RandomShift": {"max_shift": (2, 2)},
    "compositing._src.operators.BAPComposite": {"target_doy": 180},
    "compositing._src.operators.MaxNDVIComposite": {"red": 0, "nir": 1},
    "einx._src.operators.Einx": {"op": "rearrange", "pattern": "c h w -> c w h"},
    "feature._src.operators.HoughCircles": {"radii": [3, 5]},
    "geom._src.coregister.operators.VectorToRasterAgg": {"agg": "count"},
    "geom._src.operators.BowtieCorrection": {
        "detectors_per_scan": 4,
        "max_scan_angle_deg": 55.0,
        "altitude_km": 705.0,
        "aggregation_zones": ((8, 2), (4, 1)),
    },
    "geom._src.operators.CropTo": {"shape": (4, 4)},
    "geom._src.operators.CropToBounds": {"bounds": (0.0, 0.0, 1.0, 1.0)},
    "geom._src.operators.PadTo": {"shape": (8, 8)},
    "geom._src.operators.Reproject": {"dst_crs": "EPSG:4326"},
    "geom._src.operators.Resample": {"resolution": (20.0, 20.0)},
    "geom._src.operators.Resize": {"shape": (8, 8)},
    "geom._src.operators.SlidingWindow": {"size": (4, 4)},
    "geom._src.operators.Tile": {"size": (4, 4)},
    "plume._src.operators.CrossSectionalFlux": {
        "source": (500_050.0, 3_999_950.0),
        "wind_u": 1.0,
        "wind_v": 0.0,
    },
    "plume._src.operators.IMEEstimate": {"wind_speed": 1.0},
    "mask._src.operators.AltitudeMask": {"min_elev": 10.0},
    "mask._src.operators.SlopeMask": {"max_slope_deg": 30.0},
    "plume._src.operators.SBMP": {"swir1": 0, "swir2": 1},
    "indices._src.operators.AppendIndex": lambda: {
        "index_op": __import__("geotoolz.indices", fromlist=["NDVI"]).NDVI(red=0, nir=1)
    },
    "indices._src.operators.NormalizedDifference": {"a": 0, "b": 1},
    "io._src.operators.LoadFromEE": {
        "image_id": "COPERNICUS/S2/20240601T000000_20240601T000000_T29SND",
        "bounds": _BOUNDS,
        "crs": "EPSG:32629",
        "scale": 10.0,
    },
    "io._src.operators.ReadBounds": {"src": "scene.tif", "bounds": _BOUNDS},
    "io._src.operators.ReadCenterCoords": {
        "src": "scene.tif",
        "center": (500_080.0, 3_999_920.0),
        "shape": (8, 8),
    },
    "io._src.operators.ReadHDF": {"path": "scene.h5", "dataset": "radiance"},
    "io._src.operators.ReadNetCDF": {"path": "scene.nc", "variable": "radiance"},
    "io._src.operators.ReadPolygon": lambda: {
        "src": "scene.tif",
        "polygon": __import__("shapely.geometry", fromlist=["box"]).box(*_BOUNDS),
    },
    "io._src.operators.ReadTile": {"src": "scene.tif", "tile": (1, 2, 3)},
    "io._src.operators.ReadToCRS": {"src": "scene.tif", "dst_crs": "EPSG:4326"},
    "io._src.operators.ReadWindow": {"src": "scene.tif", "window": (0, 0, 8, 8)},
    "io._src.operators.WriteCOG": {"path": "out.tif"},
    "io._src.operators.WriteGeoTIFF": {"path": "out.tif"},
    "io._src.operators.WriteZarr": {"store": "out.zarr"},
    **{
        f"learn._src.operators.Pixelwise{name}": _sklearn(name)
        for name in (
            "IsolationForest",
            "IterativeImputer",
            "KMeans",
            "KNNImputer",
            "MiniBatchKMeans",
            "NMF",
            "OneClassSVM",
            "PCA",
        )
    },
    "learn._src.operators.PixelwiseLocalOutlierFactor": lambda: {
        "estimator": __import__("sklearn.neighbors").neighbors.LocalOutlierFactor(
            novelty=True
        )
    },
    "learn._src.operators.PixelwiseGMM": lambda: {
        "estimator": __import__("sklearn.mixture").mixture.GaussianMixture()
    },
    "learn._src.operators.PixelwiseIPCA": lambda: {
        "estimator": __import__("sklearn.decomposition").decomposition.IncrementalPCA()
    },
    "learn._src.operators.SklearnOp": _sklearn("PCA"),
    "mask._src.operators.BBoxMask": {"bounds": (0.0, 0.0, 1.0, 1.0)},
    "mask._src.operators.BufferMask": {"radius": 10.0},
    "mask._src.operators.CountryMask": lambda: {
        "iso_a3": "GRL",
        "source": _natural_earth_stub(),
    },
    "mask._src.operators.LandMask": lambda: {"source": _natural_earth_stub()},
    "mask._src.operators.OceanMask": lambda: {"source": _natural_earth_stub()},
    "mask._src.operators.RemoveSmallHoles": {"max_hole_area_px": 4},
    "mask._src.operators.RemoveSmallObjects": {"min_area_px": 4},
    "matched_filter._src.operators.ApplyAdaptiveMF": lambda: {"target": np.ones(3)},
    "matched_filter._src.operators.ApplyClusterMF": lambda: {"target": np.ones(3)},
    "matched_filter._src.operators.DetectionThreshold": _mf(false_alarm_rate=0.01),
    "matched_filter._src.operators.GMMClusterBackground": {"n_clusters": 2},
    "matched_filter._src.operators.MatchedFilterPixel": _mf(mean=np.zeros(3)),
    "matched_filter._src.operators.MatchedFilterSNR": _mf(amplitude=1.0),
    "matched_filter._src.operators.ValidateMFInputs": _mf(),
    "measure._src.operators.ProfileLine": {"src": (0, 0), "dst": (3, 3)},
    "normalize._src.operators.Normalize": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    "plume._src.operators.WindAdvectionCone": {
        "source": (0.0, 0.0),
        "wind_u": 1.0,
        "wind_v": 0.0,
    },
    "qa._src.operators.DecodeBitmask": {"bits": {"cloud": [3]}},
    **{
        f"qa._src.operators.{name}": {"qa_band": 0}
        for name in (
            "MaskCirrus",
            "MaskCloudShadow",
            "MaskClouds",
            "MaskSnow",
            "MaskWater",
        )
    },
    "radiometry._src.operators.ApplySRF": {
        "target_center_wavelengths": [500.0, 600.0],
        "target_fwhm": [20.0, 20.0],
        "source_wavelengths": [480.0, 520.0, 580.0, 620.0],
    },
    "radiometry._src.operators.BTFromRadiance": {"K1": 774.89, "K2": 1321.08},
    "radiometry._src.operators.ComputeSZA": {
        "center_coords": (0.0, 0.0),
        "acquisition_date": _DATE,
    },
    "radiometry._src.operators.DNToRadiance": {"gain": 0.1},
    "radiometry._src.operators.DNToReflectance": {"scale": 1e-4},
    "radiometry._src.operators.EarthSunDistanceCorrection": {"acquisition_date": _DATE},
    "radiometry._src.operators.MinMaxStretch": {"vmin": 0.0, "vmax": 1.0},
    "radiometry._src.operators.RadianceToDN": {"gain": 0.1},
    "radiometry._src.operators.RadianceToReflectance": {
        "solar_irradiance": [1900.0, 1800.0],
        "acquisition_date": _DATE,
        "sza_deg": 30.0,
    },
    "radiometry._src.operators.ReflectanceToRadiance": {
        "solar_irradiance": [1900.0, 1800.0],
        "acquisition_date": _DATE,
        "sza_deg": 30.0,
    },
    "restore._src.operators.DenoisePCA": {"n_components": 2},
    "spectral._src.operators.BandMath": {"expression": "b0 + b1"},
    "spectral._src.operators.BandRatio": {"numerator": 0, "denominator": 1},
    "spectral._src.operators.ReorderBands": {"order": [1, 0]},
    "spectral._src.operators.SelectBands": {"bands": [0, 1]},
    "spectral._src.operators.SpectralBinning": {
        "target_wavelengths": [500.0, 600.0],
        "width": 20.0,
    },
    "viz._src.operators.ApplyDiscreteColormap": {
        "mapping": {0: (0.0, 0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0, 1.0)}
    },
    "viz._src.operators.Composite": {"bands": [0, 1, 2]},
    "viz._src.operators.RGBRecipe": {"red": 0, "green": 1, "blue": 2},
    "viz._src.operators.FalseColor": {"nir": 0, "red": 1, "green": 2},
    "viz._src.operators.SWIRComposite": {"swir2": 0, "nir": 1, "red": 2},
    "viz._src.operators.TrueColor": {"red": 0, "green": 1, "blue": 2},
}

#: Operators whose constructor needs a runtime object that cannot come from a
#: config at all (a patcher, a GeoTensor, an observation model).
#: Their state round-trip is not checkable; they should be ``forbid_in_yaml``
#: (see #140) or take a config-able argument instead.
UNBUILDABLE: dict[str, str] = {
    "matched_filter._src.operators.LinearTargetFromObs": "needs an obs_model",
    "matched_filter._src.operators.NonlinearTargetFromObs": "needs an obs_model",
    "mask._src.operators._NaturalEarthMask": "private base class",
    "qa._src.operators._QAMask": "private base class",
}


def _grid() -> Any:
    from _helpers import toy_geotensor

    return toy_geotensor(np.random.default_rng(0).uniform(0.01, 1.0, (3, 16, 16)))


def _square() -> Any:
    from shapely.geometry import box

    return box(*_BOUNDS)


def _runtime(**kwargs: Callable[[], Any]) -> Callable[[], dict[str, Any]]:
    """Zero-arg spec building each runtime-object kwarg lazily."""
    return lambda: {name: make() for name, make in kwargs.items()}


#: Runtime constructor objects (GeoTensors, geometries, callables) for the
#: ``forbid_in_yaml`` / ``UNBUILDABLE`` operators. They
#: cannot come from a config, so only the checks that never serialise the
#: operator use them (``build(cls, runtime=True)``): graph mode, terminal
#: outputs, output attrs, output fill and the 4-D time-stack contract.
RUNTIME_CTOR_KWARGS: dict[str, Callable[[], dict[str, Any]]] = {
    "geom._src.operators.Georeference": _runtime(glt=_grid),
    "geom._src.operators.GeostationaryParallaxCorrect": lambda: {
        "satellite_lon_deg": 0.0
    },
    "geom._src.operators.Rasterize": _runtime(geometries=lambda: [_square()]),
    "geom._src.operators.RasterizeLike": _runtime(
        like=_grid,
        geometries=lambda: __import__("geopandas").GeoDataFrame(
            geometry=[_square()], crs="EPSG:32629"
        ),
    ),
    "geom._src.operators.ReprojectLike": _runtime(like=_grid),
    "geom._src.operators.ResampleLike": _runtime(like=_grid),
    "io._src.operators.LoadFromSTAC": lambda: {"item": object(), "asset_key": "b"},
    "io._src.operators.ReadReprojectLike": lambda: {
        "src": "scene.tif",
        "like": _grid(),
    },
    "learn._src.operators.ModelOp": _runtime(model=lambda: np.negative),
    "mask._src.operators.DistanceMask": _runtime(
        geometry=_square, distance=lambda: 10.0
    ),
    "mask._src.operators.PolygonMask": _runtime(geometry=_square),
    "matched_filter._src.operators.LinearTargetFromObs": _runtime(
        obs_model=lambda: np.exp
    ),
    "matched_filter._src.operators.NonlinearTargetFromObs": _runtime(
        obs_model=lambda: np.exp
    ),
    "measure._src.operators.RANSAC": lambda: {
        "model_class": __import__("skimage.measure").measure.LineModelND,
        "min_samples": 2,
        "residual_threshold": 1.0,
    },
    "normalize._src.operators.HistogramMatch": _runtime(reference=_grid),
    "patch_ops.BalancedSampler": lambda: {
        "labels": np.zeros((16, 16), np.int32),
        "n_per_class": 1,
        "size": (4, 4),
    },
    "patch_ops.StratifiedSample": lambda: {
        "labels": np.zeros((16, 16), np.int32),
        "target_proportions": {0: 1.0},
        "n_samples": 1,
        "size": (4, 4),
    },
    "radiometry._src.operators.IntegratedIrradiance": lambda: {
        "srf": __import__("pandas").DataFrame({"B1": [1.0]}, index=[500.0])
    },
    "viz._src.operators.AnnotatePoints": _runtime(points=lambda: [(0.0, 0.0)]),
    "viz._src.operators.AnnotatePolygons": _runtime(geometries=lambda: [_square()]),
}


def _on_grid_of(x: Any, values: np.ndarray, fill: Any = None) -> Any:
    """``values`` on ``x``'s pixel grid (a plain array when ``x`` is one)."""
    from _helpers import toy_geotensor

    if getattr(x, "transform", None) is None:
        return values
    return toy_geotensor(
        values, transform=x.transform, crs=x.crs, fill_value_default=fill
    )


def _band_like(x: Any, seed: int = 3) -> Any:
    """A single-band float field (column / DEM / intensity) on ``x``'s grid."""
    values = np.random.default_rng(seed).uniform(1.0, 100.0, np.shape(x)[-2:])
    return _on_grid_of(x, values, fill=-9999.0)


def _mask_like(x: Any) -> Any:
    """A boolean mask with a square of ``True`` pixels on ``x``'s grid."""
    h, w = np.shape(x)[-2:]
    values = np.zeros((h, w), dtype=bool)
    values[h // 4 : h // 2, w // 4 : w // 2] = True
    return _on_grid_of(x, values, fill=False)


def _keep_like(x: Any) -> Any:
    """A boolean keep-mask (``True`` = segment) with one dropped column."""
    values = np.ones(np.shape(x)[-2:], dtype=bool)
    values[:, 0] = False
    return _on_grid_of(x, values, fill=False)


def _markers_like(x: Any) -> Any:
    """Two seed labels in opposite corners on ``x``'s grid."""
    h, w = np.shape(x)[-2:]
    values = np.zeros((h, w), dtype=np.int32)
    values[1, 1] = 1
    values[h - 2, w - 2] = 2
    return _on_grid_of(x, values, fill=0)


def _labels_like(x: Any) -> Any:
    """A two-region label image on ``x``'s grid."""
    h, w = np.shape(x)[-2:]
    values = np.zeros((h, w), dtype=np.int32)
    values[: h // 2, : w // 2] = 1
    values[h // 2 :, w // 2 :] = 2
    return _on_grid_of(x, values, fill=0)


def _perturbed(x: Any) -> Any:
    """A rescaled copy of ``x`` (same grid, full shape, fill and attrs)."""
    values = np.asarray(x)
    if values.dtype.kind == "f":
        # One (H, W) factor field, so a stack's frames match the per-frame copies.
        values = values * np.random.default_rng(5).uniform(0.8, 1.2, values.shape[-2:])
    if getattr(x, "transform", None) is None:
        return values
    from georeader.geotensor import GeoTensor

    return GeoTensor(
        values,
        transform=x.transform,
        crs=x.crs,
        fill_value_default=x.fill_value_default,
        attrs=dict(x.attrs or {}),
    )


#: The second (and further) positional carriers of every two-carrier operator
#: (#141), built on the primary input's grid: ``op(x, *MULTI_INPUTS[key](x))``.
#: Optional carriers are included, so every input is exercised.
MULTI_INPUTS: dict[str, Callable[[Any], tuple[Any, ...]]] = {
    "augment._src.operators.CutMix": lambda x: (_perturbed(x),),
    "geom._src.operators.OpticalFlowILK": lambda x: (_perturbed(x),),
    "geom._src.operators.OpticalFlowTVL1": lambda x: (_perturbed(x),),
    "geom._src.operators.PhaseAlign": lambda x: (_perturbed(x),),
    "mask._src.operators.AltitudeMask": lambda x: (_band_like(x),),
    "mask._src.operators.ApplyMask": lambda x: (_mask_like(x),),
    "mask._src.operators.SlopeMask": lambda x: (_band_like(x),),
    "measure._src.operators.RegionProps": lambda x: (_band_like(x),),
    "plume._src.operators.CrossSectionalFlux": lambda x: (_mask_like(x),),
    "plume._src.operators.IMEEstimate": lambda x: (_mask_like(x),),
    "plume._src.operators.PlumeColumnStats": lambda x: (_band_like(x),),
    "plume._src.operators.PlumeFootprint": lambda x: (_band_like(x),),
    "plume._src.operators.PlumeQNDFeatures": lambda x: (
        _band_like(x),
        _band_like(x, seed=4),
    ),
    "plume._src.operators.SBMP": lambda x: (_perturbed(x),),
    "segment._src.operators.Felzenszwalb": lambda x: (_keep_like(x),),
    "segment._src.operators.MarkBoundaries": lambda x: (_labels_like(x),),
    "segment._src.operators.Quickshift": lambda x: (_keep_like(x),),
    "segment._src.operators.RandomWalker": lambda x: (_markers_like(x),),
    "segment._src.operators.SLIC": lambda x: (_keep_like(x),),
    "segment._src.operators.Watershed": lambda x: (
        _markers_like(x),
        _keep_like(x),
    ),
}

#: ``{check: {operator key: (issue, expected exception)}}`` — strict xfails
#: removed as fixed. The exception type pins each case to its documented
#: failure mode, so a different failure is reported instead of swallowed.
Known = tuple[str, type[BaseException]]

KNOWN_FAILURES: dict[str, dict[str, Known]] = {
    "round_trip": {},
    "config_is_json": {},
    "config_keys": {},
    "graph_mode": {},
}

_CLASSES = all_operator_classes()


def _params(check: str, classes: list[type] | None = None) -> list[Any]:
    known = KNOWN_FAILURES[check]
    out = []
    for cls in _CLASSES if classes is None else classes:
        key = _key(cls)
        marks = []
        if key in known:
            reason, raises = known[key]
            marks.append(pytest.mark.xfail(reason=reason, raises=raises, strict=True))
        out.append(pytest.param(cls, id=key, marks=marks))
    return out


def build(cls: type, *, runtime: bool = False) -> Operator:
    """Construct ``cls`` from ``CTOR_KWARGS`` or with no arguments.

    With ``runtime=True`` (checks that never serialise the operator),
    ``RUNTIME_CTOR_KWARGS`` supplies the runtime objects first. Otherwise
    skips when the class needs runtime objects (``UNBUILDABLE``) or is a
    ``forbid_in_yaml`` class with no entry.
    """
    key = _key(cls)
    if runtime and key in RUNTIME_CTOR_KWARGS:
        return cls(**RUNTIME_CTOR_KWARGS[key]())
    if key in UNBUILDABLE:
        pytest.skip(UNBUILDABLE[key])
    spec = CTOR_KWARGS.get(key, {})
    kwargs = spec() if callable(spec) else spec
    try:
        return cls(**kwargs)
    except TypeError:
        if cls.forbid_in_yaml and key not in CTOR_KWARGS:
            pytest.skip("forbid_in_yaml operator needs runtime constructor args")
        raise


def test_every_operator_is_classified() -> None:
    """New operators must be buildable, in ``CTOR_KWARGS``, or listed as unbuildable.

    ``forbid_in_yaml`` / ``UNBUILDABLE`` classes that need runtime objects
    must be in ``RUNTIME_CTOR_KWARGS`` (private bases excepted). Keeps the
    contract tests below from silently skipping new classes.
    """
    missing = []
    for cls in _CLASSES:
        key = _key(cls)
        if key in CTOR_KWARGS or key in RUNTIME_CTOR_KWARGS:
            continue
        if cls.__name__.startswith("_"):
            continue
        try:
            cls()
        except (TypeError, ValueError):
            missing.append(key)
    assert missing == [], "add constructor kwargs to (RUNTIME_)CTOR_KWARGS"


def _patch_ops_operators() -> list[type]:
    """The geopatcher operators ``geotoolz.patch_ops`` re-exports as its own."""
    try:
        import geotoolz.patch_ops as patch_ops
    except ImportError:  # pragma: no cover - the [patch] extra is missing
        return []
    return [
        obj
        for name in patch_ops.__all__
        if isinstance(obj := getattr(patch_ops, name), type)
        and issubclass(obj, Operator)
        and not obj.__module__.startswith("geotoolz")
    ]


@pytest.mark.parametrize(
    "cls",
    [pytest.param(c, id=_key(c)) for c in [*_CLASSES, *_patch_ops_operators()]],
)
def test_constructors_are_keyword_only(cls: type) -> None:
    """Every operator constructor takes its parameters by keyword only.

    ``__init__`` must start with a bare ``*`` -- no exception for the
    wrapped estimator / model / patcher -- so YAML / Hydra configs and
    call sites are unambiguous and parameters can be renamed or reordered
    safely. ``*args`` / ``**kwargs`` pass-throughs (inherited base
    constructors) are allowed.
    """
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    positional = [
        p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == [], (
        f"{cls.__qualname__}.__init__ takes {positional} positionally"
    )


#: Retired constructor spellings -> the package-wide name (see the
#: vocabulary table in packages/geotoolz/AGENTS.md).
RETIRED_PARAMS: dict[str, str] = {
    "fill": "fill_value",
    "random_state": "seed",
    "window_size": "window",
    "patch_size": "window",
    "min_area": "min_area_px",
    "min_size": "min_area_px",
    "channel_axis": "axis",
    "wavelengths_nm": "wavelengths",
}

#: Documented exceptions: (class key, parameter) pairs whose name is kept
#: because it is a different concept from the retired spelling's target.
VOCABULARY_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Gaussian kernel *width* (a float scale), not a window side length.
        ("segment._src.operators.Quickshift", "kernel_size"),
    }
)


@pytest.mark.parametrize("cls", [pytest.param(c, id=_key(c)) for c in _CLASSES])
def test_constructor_vocabulary(cls: type) -> None:
    """Constructors use the package-wide parameter spellings.

    No retired alias (``fill``, ``random_state``, ``window_size`` ...), no
    ``*_idx`` band-index twin, no ``kernel_size`` window, and ``axis`` is
    only ever an integer band position (reductions use ``reduce_axes``,
    orientations ``direction``).
    """
    key = _key(cls)
    params = inspect.signature(cls.__init__).parameters
    for name, param in params.items():
        if (key, name) in VOCABULARY_EXCEPTIONS:
            continue
        assert name not in RETIRED_PARAMS, (
            f"{key}: rename {name!r} -> {RETIRED_PARAMS[name]!r}"
        )
        assert not name.endswith("_idx"), f"{key}: {name!r} -- use one BandRef name"
        assert name != "kernel_size", f"{key}: 'kernel_size' -> 'window'"
        if name == "axis" and param.default is not inspect.Parameter.empty:
            default = param.default
            assert default is None or isinstance(default, int), (
                f"{key}: axis={default!r} -- 'axis' is a band position; use "
                "'reduce_axes' for reductions and 'direction' for orientations"
            )


def _importable(key: str) -> bool:
    """Whether the module behind a table key imports (optional extras)."""
    import importlib

    parts = key.split(".")[:-1]
    try:
        importlib.import_module(".".join(["geotoolz", *parts]))
    except ImportError:
        return False
    return True


def test_tables_name_real_operators() -> None:
    """Table keys name operators; entries for missing extras are ignored."""
    keys = {_key(c) for c in _CLASSES}
    tables = [
        CTOR_KWARGS,
        RUNTIME_CTOR_KWARGS,
        UNBUILDABLE,
        INTERMEDIATE_OUTPUTS,
        ATTRS_KNOWN_FAILURES,
        FILL_KNOWN_FAILURES,
        TIME_STACK_KNOWN_FAILURES,
        *KNOWN_FAILURES.values(),
    ]
    stale = sorted(k for k in {k for t in tables for k in t} - keys if _importable(k))
    assert stale == []


def _is_nested_operator(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(map(_is_nested_operator, value))
    return isinstance(value, dict) and set(value) == {"class", "config"}


def _same(a: Any, b: Any) -> bool:
    """Equality that treats NaN as equal to NaN."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a):
        return bool(np.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        same_kind = type(a) is type(b)
        return same_kind and len(a) == len(b) and all(map(_same, a, b))
    return a == b


@pytest.mark.parametrize("cls", _params("round_trip"))
def test_state_round_trip(cls: type) -> None:
    """``Operator.from_state(op.state)`` rebuilds an equal operator."""
    if cls.forbid_in_yaml:
        state = {"module": cls.__module__, "class": cls.__name__, "config": {}}
        with pytest.raises(RuntimeError, match="forbid_in_yaml"):
            Operator.from_state(state)
        return
    op = build(cls)
    state = json.loads(json.dumps(op.state))
    if any(_is_nested_operator(v) for v in op.get_config().values()):
        # Containers emit nested debug payloads; pipekit refuses to rebuild
        # them through from_state (use a YAML / Hydra loader instead).
        with pytest.raises(RuntimeError, match="non-primitive"):
            Operator.from_state(state)
        return
    clone = Operator.from_state(state)
    assert type(clone) is cls
    assert _same(clone.get_config(), op.get_config())


@pytest.mark.parametrize("cls", _params("config_is_json"))
def test_get_config_is_json(cls: type) -> None:
    """``get_config()`` serialises as strict JSON (no NaN / inf)."""
    op = build(cls)
    json.dumps(op.get_config(), allow_nan=False)


@pytest.mark.parametrize(
    "cls", _params("config_keys", [c for c in _CLASSES if not c.forbid_in_yaml])
)
def test_get_config_keys_are_constructor_params(cls: type) -> None:
    """Every ``get_config`` key is a constructor parameter."""
    op = build(cls)
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    assert set(op.get_config()) <= set(params) - {"self"}


def _is_boilerplate_get_config(cls: type) -> bool:
    """``get_config`` returns exactly ``{p: self.p}`` for every ctor param."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(cls.get_config))).body[0]
    body = [
        s
        for s in fn.body  # ty: ignore[unresolved-attribute]
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    ret = body[0].value
    if not isinstance(ret, ast.Dict):
        return False
    keys = []
    for key, value in zip(ret.keys, ret.values, strict=True):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
            and value.attr == key.value
        ):
            return False
        keys.append(key.value)
    params = inspect.signature(cls.__init__).parameters.values()
    names = [p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return sorted(keys) == sorted(n for n in names if n != "self")


def test_no_boilerplate_get_config_overrides() -> None:
    """Overrides that restate ``ConfigMixin``'s auto-derived config drift (#138).

    Store constructor kwargs under the same attribute name and let
    ``ConfigMixin`` derive the config. An override is only needed to
    coerce values or to shadow a parent's custom ``get_config``.
    """
    offenders = []
    for cls in _CLASSES:
        if "get_config" not in vars(cls) or not _is_boilerplate_get_config(cls):
            continue
        shadows_parent = any(
            "get_config" in vars(base)
            for base in cls.__mro__[1 : cls.__mro__.index(Operator)]
        )
        if not shadows_parent:
            offenders.append(_key(cls))
    assert offenders == []


def _n_inputs(op: Operator) -> int:
    """Graph inputs to feed: the required positional parameters of ``_apply``.

    An ``_apply`` whose only positional parameters are optional (pass-through
    validators, optional-input ops) or variadic still takes one input; zero is
    reserved for ``_apply`` methods with no positional parameters at all.
    """
    params = list(inspect.signature(op._apply).parameters.values())
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    required = sum(1 for p in params if p.kind in positional and p.default is p.empty)
    if required:
        return required
    return int(any(p.kind in positional or p.kind is p.VAR_POSITIONAL for p in params))


def _graph_arity(op: Operator) -> int:
    """Graph inputs to wire: every carrier of a multi-input operator (#141)."""
    key = _key(type(op))
    if key in MULTI_INPUTS:
        return 1 + len(MULTI_INPUTS[key](_toy_inputs()[0]))
    if key in NARY_INPUTS:
        return len(NARY_INPUTS[key]())
    return _n_inputs(op)


@pytest.mark.parametrize("cls", _params("graph_mode"))
def test_graph_mode(cls: type) -> None:
    """Calling an operator on ``Input`` nodes returns a ``Node``.

    Multi-input operators are called on one ``Input`` per carrier (#141).
    Graph construction never serialises the operator, so ``forbid_in_yaml``
    and ``UNBUILDABLE`` classes are built from ``RUNTIME_CTOR_KWARGS``.
    """
    op = build(cls, runtime=True)
    n = _graph_arity(op)
    if n == 0:
        pytest.skip("input-less operator (source)")
    node = op(*(Input(f"x{i}") for i in range(n)))
    assert isinstance(node, Node)
    assert node.operator is op
    assert len(node.parents) == n


# ---------------------------------------------------------------------------
# Multi-input operators (#141)
# ---------------------------------------------------------------------------


def _two_frames() -> list[Any]:
    scene = _scene()
    return [scene, _perturbed(scene)]


#: Inputs of the N-ary reducers (#141), wired one ``Input`` each in a graph.
NARY_INPUTS: dict[str, Callable[[], list[Any]]] = {
    "compositing._src.operators.BAPComposite": lambda: [
        (frame, {"doy": 170 + t}) for t, frame in enumerate(_two_frames())
    ],
    "compositing._src.operators.BlendMatched": _two_frames,
    "compositing._src.operators.CloudFreeComposite": lambda: [
        (frame, _mask_like(frame)) for frame in _two_frames()
    ],
    "compositing._src.operators.MaxNDVIComposite": _two_frames,
    "compositing._src.operators.MedianComposite": _two_frames,
    "compositing._src.operators.MinCloudComposite": lambda: [
        (frame, _mask_like(frame)) for frame in _two_frames()
    ],
    "compositing._src.operators.StackMatched": _two_frames,
    "mask._src.operators.CombineMasks": lambda: [
        _mask_like(_scene()),
        _keep_like(_scene()),
    ],
}


def _multi_input_classes() -> list[Any]:
    return [
        pytest.param(cls, id=_key(cls))
        for cls in _CLASSES
        if _key(cls) in MULTI_INPUTS or _key(cls) in NARY_INPUTS
    ]


#: The primary carrier of a two-carrier operator in the graph checks, when
#: not the 3-band ``_scene()``: label maps for the per-instance tables, a
#: single-band field for the plume quantifiers and the 1-band segmenters.
GRAPH_PRIMARY: dict[str, Callable[[], Any]] = {
    "measure._src.operators.RegionProps": lambda: _labels(),
    "plume._src.operators.CrossSectionalFlux": lambda: _band_like(_scene()),
    "plume._src.operators.IMEEstimate": lambda: _band_like(_scene()),
    "plume._src.operators.PlumeColumnStats": lambda: _labels(),
    "plume._src.operators.PlumeFootprint": lambda: _labels(),
    "plume._src.operators.PlumeQNDFeatures": lambda: _labels(),
    "segment._src.operators.RandomWalker": lambda: _band_like(_scene()),
    "segment._src.operators.Watershed": lambda: _band_like(_scene()),
}


def _graph_values(cls: type) -> list[Any]:
    """Real inputs for a multi-input operator: one value per graph ``Input``."""
    key = _key(cls)
    if key in NARY_INPUTS:
        return NARY_INPUTS[key]()
    primary = GRAPH_PRIMARY.get(key, _scene)()
    return [primary, *MULTI_INPUTS[key](primary)]


def _assert_same_output(got: Any, want: Any) -> None:
    import pandas as pd

    if isinstance(want, pd.DataFrame):
        pd.testing.assert_frame_equal(got, want)
    elif isinstance(want, dict):
        assert got.keys() == want.keys()
        for name in want:
            _assert_same_output(got[name], want[name])
    elif isinstance(want, tuple | list):
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _assert_same_output(g, w)
    else:
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_every_multi_input_operator_is_classified() -> None:
    """An operator whose ``_apply`` takes several carriers is in a table.

    Several positional ``_apply`` parameters (or ``*frames``) mark a
    multi-input operator, which must be wired and grid-checked below.
    """
    missing = []
    for cls in _CLASSES:
        key = _key(cls)
        if key in MULTI_INPUTS or key in NARY_INPUTS:
            continue
        params = list(inspect.signature(cls._apply).parameters.values())[1:]
        if any(p.kind is p.VAR_KEYWORD for p in params):
            continue  # generic pass-through (the io source / sink bases)
        positional = [
            p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        variadic = any(p.kind is p.VAR_POSITIONAL for p in params)
        if len(positional) > 1 or (variadic and positional == []):
            missing.append(key)
    assert sorted(missing) == sorted(POSITIONAL_PAIR_OPERATORS), (
        "add the operator to MULTI_INPUTS / NARY_INPUTS"
    )


#: Operators that already took two positional carriers before #141 and are
#: covered by their own test modules (coregistration, dNBR, overlays, the
#: matched-filter stages that take a fitted background).
POSITIONAL_PAIR_OPERATORS: tuple[str, ...] = (
    "geom._src.coregister.operators.PointCloudToRaster",
    "geom._src.coregister.operators.PointsToRaster",
    "geom._src.coregister.operators.RasterToPointCloud",
    "geom._src.coregister.operators.RasterToPoints",
    "geom._src.coregister.operators.RasterToRasterLike",
    "geom._src.coregister.operators.VectorToRasterAgg",
    "indices._src.operators.dNBR",
    "matched_filter._src.operators.ApplyAdaptiveMF",
    "matched_filter._src.operators.ApplyClusterMF",
    "viz._src.operators.Overlay",
)


@pytest.mark.parametrize("cls", _multi_input_classes())
def test_multi_input_operator_runs_in_a_graph(cls: type) -> None:
    """``op(Input("x0"), Input("x1"), ...)`` in a ``Graph`` equals the eager call.

    The multi-input convention (#141): every carrier is a positional
    argument, so a graph wires each one from its own ``Input``.
    """
    from pipekit import Graph

    values = _graph_values(cls)
    inputs = {f"x{i}": Input(f"x{i}") for i in range(len(values))}
    graph = Graph(inputs=inputs, outputs={"out": _build_seeded(cls)(*inputs.values())})
    got = graph(**{name: v for name, v in zip(inputs, values, strict=True)})["out"]
    _assert_same_output(got, _build_seeded(cls)(*values))


def _off_grid(x: Any) -> Any:
    """``x`` shifted by one pixel: same shape, another grid."""
    from affine import Affine
    from georeader.geotensor import GeoTensor

    return GeoTensor(
        np.asarray(x),
        transform=x.transform * Affine.translation(1, 0),
        crs=x.crs,
        fill_value_default=x.fill_value_default,
        attrs=dict(x.attrs or {}),
    )


@pytest.mark.parametrize("cls", _multi_input_classes())
def test_multi_input_operator_rejects_an_off_grid_carrier(cls: type) -> None:
    """A second carrier on another grid raises a ``ValueError`` naming the op.

    Carriers are combined pixel by pixel, so a mask / DEM / reference /
    frame shifted by one pixel would silently give a wrong answer (#141).
    """
    values = _graph_values(cls)
    last = values[-1]
    if isinstance(last, tuple):
        values[-1] = (_off_grid(last[0]), last[1])
    else:
        values[-1] = _off_grid(last)
    with pytest.raises(ValueError, match=rf"{cls.__name__}: the .* pixel grid"):
        _build_seeded(cls)(*values)


@pytest.mark.parametrize("cls", _multi_input_classes())
def test_carriers_are_not_constructor_kwargs(cls: type) -> None:
    """No constructor parameter takes a carrier; the operator round-trips.

    The old carrier kwargs (``plume_mask``, ``dem``, ``pool``, ...) are gone
    (#141): constructors hold scalars and configuration only.
    """
    removed = {
        "albedo",
        "column",
        "dem",
        "enhancement",
        "intensity_image",
        "label_img",
        "markers",
        "plume_mask",
        "pool",
        "reference",
        "reference_scene",
    }
    from geotoolz.mask import ApplyMask

    params = set(inspect.signature(cls.__init__).parameters)
    assert not params & removed
    # ApplyMask(mask=...) takes a mask-producing *operator*, not a carrier.
    assert "mask" not in params or cls is ApplyMask
    op = build(cls)
    if not cls.forbid_in_yaml:
        clone = Operator.from_state(json.loads(json.dumps(op.state)))
        assert _same(clone.get_config(), op.get_config())


def _scene() -> Any:
    from _helpers import toy_geotensor

    values = np.random.default_rng(0).uniform(0.0, 1.0, (3, 16, 16))
    return toy_geotensor(values, attrs={"band_names": ["b0", "b1", "b2"]})


def _labels() -> Any:
    from _helpers import toy_geotensor

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[2:5, 7:10] = 2
    return toy_geotensor(labels, fill_value_default=0)


def _tiles() -> Any:
    from geotoolz.geom import Tile

    return list(Tile(size=(8, 8))(_scene()))


def _reload_cases() -> list[Any]:
    from affine import Affine

    import geotoolz as gz

    scene_transform = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    cases = [
        (gz.augment.GaussianNoise(sigma=(0.0, 0.1), seed=0), _scene),
        (gz.augment.AtmosphericHaze(intensity=(0.0, 0.2), seed=0), _scene),
        (gz.augment.SimulatedClouds(seed=0), _scene),
        (gz.augment.BrightnessJitter(factor=(0.9, 1.1), seed=0), _scene),
        (gz.augment.BandJitter(groups={"vis": ["b0", "b1"]}, seed=0), _scene),
        (gz.radiometry.PercentileClip(reduce_axes=(-2, -1)), _scene),
        (gz.qa.DecodeBitmask(bits={"low": [0], "high": [1]}), _labels),
        (gz.segment.MergeNearbyInstances(classes={1: 0, 2: 0}), _labels),
        (
            gz.geom.Stitch(target_transform=scene_transform, target_shape=(16, 16)),
            _tiles,
        ),
    ]
    return [pytest.param(op, make, id=type(op).__name__) for op, make in cases]


@pytest.mark.parametrize(("op", "make_input"), _reload_cases())
def test_reloaded_operator_applies_identically(
    op: Operator, make_input: Callable[[], Any]
) -> None:
    """Tuple / mapping configs reload into operators that still run (#139)."""
    clone = Operator.from_state(json.loads(json.dumps(op.state)))
    np.testing.assert_array_equal(
        np.asarray(clone(make_input())), np.asarray(op(make_input()))
    )


#: Operators whose non-carrier output is meant to feed a downstream step:
#: fan-outs returning a list of carriers, and matched-filter stages that
#: hand a statistic to the next stage. Not terminal by design.
INTERMEDIATE_OUTPUTS: dict[str, str] = {
    "geom._src.operators.SlidingWindow": "fan-out: list of tiles",
    "geom._src.operators.Tile": "fan-out: list of tiles",
    "spectral._src.operators.SplitBands": "fan-out: list of bands",
    "einx._src.operators.PerBandReduce": "per-band statistic vector",
    "matched_filter._src.operators.AdaptiveWindowBackground": "MF stage",
    "matched_filter._src.operators.EstimateCovEmpirical": "MF stage",
    "matched_filter._src.operators.EstimateCovLowRank": "MF stage",
    "matched_filter._src.operators.EstimateCovShrunk": "MF stage",
    "matched_filter._src.operators.EstimateMean": "MF stage",
    "matched_filter._src.operators.GMMClusterBackground": "MF stage",
    "matched_filter._src.operators.StreamingBackground": "MF stage",
    "matched_filter._src.operators.LinearTargetFromObs": "MF stage: target",
    "matched_filter._src.operators.NonlinearTargetFromObs": "MF stage: target",
    "patch_ops.BalancedSampler": "fan-out: list of chips",
    "patch_ops.StratifiedSample": "fan-out: list of chips",
}


def _toy_inputs() -> list[Any]:
    from _helpers import toy_geotensor

    rng = np.random.default_rng(0)
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:6] = 1
    return [
        toy_geotensor(
            rng.uniform(0.01, 1.0, (3, 16, 16)),
            attrs={"band_names": ["b0", "b1", "b2"]},
        ),
        toy_geotensor(rng.uniform(0.01, 1.0, (16, 16))),
        toy_geotensor(labels, fill_value_default=0),
        toy_geotensor(labels.astype(np.uint8), fill_value_default=0),
        np.full(3, 0.5),  # a single 1-D spectrum (per-pixel MF ops)
    ]


def _is_carrier(value: Any) -> bool:
    from georeader.geotensor import GeoTensor

    return isinstance(value, GeoTensor) or (
        isinstance(value, np.ndarray) and value.ndim >= 2
    )


def _single_carrier_in(op: Operator) -> bool:
    """One primary input: a single-input operator, or a two-carrier one whose
    further carriers ``MULTI_INPUTS`` builds on the primary's grid (#141)."""
    return _n_inputs(op) == 1 or _key(type(op)) in MULTI_INPUTS


def _call(op: Operator, x: Any) -> Any:
    """``op(x)``, plus the further carriers of a two-carrier operator."""
    extra = MULTI_INPUTS.get(_key(type(op)))
    return op(x) if extra is None else op(x, *extra(x))


@pytest.mark.parametrize("cls", _params("graph_mode"))
def test_non_carrier_outputs_are_terminal(cls: type) -> None:
    """An operator that returns a non-carrier is ``_terminal`` (#142).

    ``Sequential`` then rejects it mid-pipeline at construction instead
    of failing downstream with an unrelated error. Runs the operator on
    a few toy scenes; skips when none is a valid input.
    """
    op = build(cls, runtime=True)
    if op._terminal:
        # Already terminal; also keeps sinks from writing files here.
        return
    if _key(cls) in INTERMEDIATE_OUTPUTS or not _single_carrier_in(op):
        pytest.skip("intermediate output or not a single-input operator")
    for scene in _toy_inputs():
        try:
            out = _call(op, scene)
        except Exception:
            continue
        if not _is_carrier(out):
            assert op._terminal, f"returns {type(out).__name__}"
        return
    pytest.skip("no toy input is valid for this operator")


def test_phase_align_is_terminal_only_without_apply() -> None:
    """``PhaseAlign(apply=False)`` returns a shift tuple (#142 review)."""
    from pipekit import Identity, Sequential

    from geotoolz.geom import PhaseAlign

    reference = _toy_inputs()[1]
    shifted = PhaseAlign()
    assert not shifted._terminal
    Sequential([shifted, Identity()])
    raw = PhaseAlign(apply=False)
    assert isinstance(raw(reference, reference), tuple)
    with pytest.raises(TypeError, match="terminal"):
        Sequential([raw, Identity()])


@pytest.mark.parametrize(
    ("cls_path", "kwargs"),
    [
        ("compositing.MedianComposite", {"return_count": True}),
        ("compositing.MaxNDVIComposite", {"red": 0, "nir": 1, "return_index": True}),
        ("compositing.CloudFreeComposite", {"return_count": True}),
        ("compositing.BAPComposite", {"target_doy": 180, "return_score": True}),
        ("compositing.MinCloudComposite", {"return_count": True}),
    ],
)
def test_tuple_returning_composites_are_terminal_only_when_flagged(
    cls_path: str, kwargs: dict[str, Any]
) -> None:
    from pipekit import Identity, Sequential

    import geotoolz as gz

    module, name = cls_path.split(".")
    cls = getattr(getattr(gz, module), name)
    flag = next(k for k in kwargs if k.startswith("return_"))
    Sequential([cls(**{**kwargs, flag: False}), Identity()])
    with pytest.raises(TypeError, match="terminal"):
        Sequential([cls(**kwargs), Identity()])


def _spectral_scene(wavelengths: list[float], *, seed: int = 0, **extra: Any) -> Any:
    """A float scene with one ``band_names`` / ``descriptions`` / ``wavelengths``
    entry per band."""
    from _helpers import toy_geotensor

    n = len(wavelengths)
    names = [f"b{i}" for i in range(n)]
    values = np.random.default_rng(seed).uniform(0.01, 1.0, (n, 16, 16))
    return toy_geotensor(
        values,
        attrs={
            "band_names": names,
            "descriptions": list(names),
            "wavelengths": list(wavelengths),
            "sensor": "toy",
            **extra,
        },
    )


def _attributed_inputs() -> list[Any]:
    """Toy inputs whose per-band attrs are consistent with their band count.

    Band counts and names are chosen so that operators with sensor-style
    defaults (indices, SRFs, QA registries, list-input composites) run.
    """
    from _helpers import toy_geotensor

    rng = np.random.default_rng(0)
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:6] = 1
    qa_names = ["QA60", "SCL", "QA_PIXEL", "state_1km"]
    return [
        _spectral_scene([490.0, 560.0, 665.0]),
        _spectral_scene(
            [
                *(443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0),
                *(945.0, 1375.0, 1610.0, 2190.0),
            ],
            sza_deg=30.0,
        ),
        _spectral_scene([500.0, 600.0]),
        _spectral_scene([495.0, 505.0, 595.0, 605.0]),
        toy_geotensor(
            rng.uniform(0.01, 1.0, (16, 16)),
            attrs={"band_names": ["b0"], "sensor": "toy"},
        ),
        toy_geotensor(labels, fill_value_default=0, attrs={"band_names": ["labels"]}),
        toy_geotensor(
            labels.astype(np.uint8), fill_value_default=0, attrs={"band_names": ["qa"]}
        ),
        toy_geotensor(
            rng.integers(0, 2**12, (4, 16, 16)).astype(np.uint16),
            fill_value_default=0,
            attrs={"band_names": qa_names, "descriptions": list(qa_names)},
        ),
        [
            _spectral_scene([490.0, 560.0, 665.0], seed=1),
            _spectral_scene([490.0, 560.0, 665.0], seed=2),
        ],
    ]


def _takes_sequence(op: Operator) -> bool:
    """Whether ``op`` reduces a sequence / mapping of carriers.

    N-ary reducers (``_apply(self, *frames)``, #141) also take one list.
    """
    params = list(inspect.signature(op._apply).parameters.values())
    if params and params[0].kind is inspect.Parameter.VAR_POSITIONAL:
        return True
    annotation = str(params[0].annotation) if params else ""
    return any(tag in annotation for tag in ("Sequence", "Mapping", "list["))


#: Strict xfails for ``test_output_attrs_are_fresh_and_consistent``, owned by
#: later branches of the #112 stack.
ATTRS_KNOWN_FAILURES: dict[str, Known] = {}


def _attrs_params() -> list[Any]:
    out = []
    for cls in _CLASSES:
        key = _key(cls)
        marks = []
        if key in ATTRS_KNOWN_FAILURES:
            reason, raises = ATTRS_KNOWN_FAILURES[key]
            marks.append(pytest.mark.xfail(reason=reason, raises=raises, strict=True))
        out.append(pytest.param(cls, id=key, marks=marks))
    return out


@pytest.mark.parametrize("cls", _attrs_params())
def test_output_attrs_are_fresh_and_consistent(cls: type) -> None:
    """Outputs never alias the input's ``attrs`` nor carry stale band keys (#144).

    Runs every buildable single-input operator on toy scenes carrying
    ``band_names`` / ``descriptions`` / ``wavelengths``; for each GeoTensor
    output, ``out.attrs is not gt.attrs`` and every per-band attrs list has
    one entry per output band.
    """
    op = build(cls, runtime=True)
    if op._terminal or not _single_carrier_in(op):
        pytest.skip("terminal or not a single-input operator")
    ran = False
    takes_sequence = _takes_sequence(op)
    for scene in _attributed_inputs():
        # A bare carrier handed to a sequence operator iterates as band
        # slices that alias its attrs; that is not a supported call.
        if isinstance(scene, list) != takes_sequence:
            continue
        try:
            out = _call(op, scene)
        except Exception:
            continue
        ran = True
        assert_fresh_consistent_attrs(scene, out)
    if not ran:
        pytest.skip("no toy input is valid for this operator")


#: Operators that fill nodata gaps by design: input fill pixels become
#: valid output, so only the dtype half of the fill contract applies.
GAP_FILLERS: frozenset[str] = frozenset(
    f"restore._src.operators.{name}"
    for name in (
        "GapFillIDW",
        "GapFillInpaintBiharmonic",
        "GapFillLaplacian",
        "GapFillNearest",
    )
) | {
    # Swath-edge nodata (e.g. VIIRS bowtie-deleted rows) is refilled from
    # the overlapping scan that sees the same ground; the toy scene's
    # corner fill pixels sit in that overlap.
    "geom._src.operators.BowtieCorrection",
    # Pastes a random donor rectangle over the input by design; when the
    # rectangle covers the toy scene's corner fill pixels, the donor's
    # valid pixels replace them (a 2-3% chance per unseeded draw).
    "augment._src.operators.CutMix",
}

#: Strict xfails for ``test_output_fill_matches_dtype`` (none open).
FILL_KNOWN_FAILURES: dict[str, Known] = {}


def _fill_params() -> list[Any]:
    out = []
    for cls in _CLASSES:
        key = _key(cls)
        marks = []
        if key in FILL_KNOWN_FAILURES:
            reason, raises = FILL_KNOWN_FAILURES[key]
            marks.append(pytest.mark.xfail(reason=reason, raises=raises, strict=True))
        out.append(pytest.param(cls, id=key, marks=marks))
    return out


def _with_fill_pixels(scene: Any) -> Any:
    """``scene`` (or each scene of a list) with nodata written into two pixels."""
    from _helpers import toy_geotensor

    if isinstance(scene, list):
        return [_with_fill_pixels(s) for s in scene]
    return toy_geotensor(
        np.asarray(scene),
        fill_value_default=scene.fill_value_default,
        attrs=scene.attrs,
        with_fill_pixels=True,
    )


@pytest.mark.parametrize("cls", _fill_params())
def test_output_fill_matches_dtype(cls: type) -> None:
    """Output fill values follow the output's dtype and meaning (#146).

    Runs every buildable single-input operator on toy scenes whose corner
    pixels hold the input's fill value, and checks each GeoTensor output
    with `geotoolz._src.contract.assert_fill_matches_dtype`.
    """
    op = build(cls, runtime=True)
    if op._terminal or not _single_carrier_in(op):
        pytest.skip("terminal or not a single-input operator")
    ran = False
    takes_sequence = _takes_sequence(op)
    for scene in _attributed_inputs():
        if isinstance(scene, list) != takes_sequence:
            continue
        scene = _with_fill_pixels(scene)
        try:
            out = _call(op, scene)
        except Exception:
            continue
        ran = True
        src = scene[0] if isinstance(scene, list) else scene
        sources = geotensors(scene)
        for gt in geotensors(out):
            if any(gt is s for s in sources):
                continue
            assert_fill_matches_dtype(
                src,
                gt,
                gap_filler=_key(cls) in GAP_FILLERS,
                nodata=fill_pixel_mask(src.shape),
            )
    if not ran:
        pytest.skip("no toy input is valid for this operator")


# ---------------------------------------------------------------------------
# 4-D (T, C, H, W) time stacks (#147)
# ---------------------------------------------------------------------------

#: Operators whose per-pixel draws depend on the input's full shape, so a
#: stack cannot reproduce the per-frame draws; only the output shape is
#: checked.
STOCHASTIC_PER_PIXEL: frozenset[str] = frozenset(
    {
        "augment._src.operators.GaussianNoise",
        "augment._src.operators.SpeckleNoise",
    }
)

#: Operators that fit statistics (a per-band mean, a background covariance,
#: a stretch range, principal components) and deliberately pool them over
#: every frame of a stack -- one fit for the whole time series -- so their
#: values differ from per-frame fits; only the output shape is checked.
POOLED_STATISTICS: dict[str, str] = {
    "augment._src.operators.SimulatedClouds": "cloud brightness of the stack",
    "matched_filter._src.operators.ColumnEnhancement": "one background fit",
    "normalize._src.operators.HistogramMatch": "one source CDF per band",
    "normalize._src.operators.HistogramStretch": "per-band percentiles",
    "normalize._src.operators.ZeroOne": "per-band min / max",
    "restore._src.operators.DenoisePCA": "one PCA basis",
    "restore._src.operators.MNF": "one PCA basis",
}

#: Operators whose output depends only on the carrier's grid (a rasterised
#: geometry) or on a static second carrier (a DEM), so one ``(H, W)`` mask
#: serves every frame of a stack.
TIME_INVARIANT: frozenset[str] = frozenset(
    f"mask._src.operators.{name}"
    for name in (
        "BBoxMask",
        "CountryMask",
        "DistanceMask",
        "LandMask",
        "OceanMask",
        "PolygonMask",
        "AltitudeMask",
        "SlopeMask",
    )
) | {"geom._src.operators.Rasterize", "geom._src.operators.RasterizeLike"}

#: learn estimators in their default ``mode="pixel"`` treat every
#: non-spatial axis as a feature, so a ``(T, C, H, W)`` stack is fitted on
#: ``T * C`` features per pixel and returns one ``(k, H, W)`` map -- by
#: design, not per frame (``mode="pixel_time"`` gives ``(T, k, H, W)``; see
#: ``test_learn``). Only the output grid is checked.
STACK_AS_FEATURES: frozenset[str] = frozenset(
    {
        *(
            f"learn._src.operators.Pixelwise{name}"
            for name in (
                "GMM",
                "IPCA",
                "IsolationForest",
                "IterativeImputer",
                "KMeans",
                "KNNImputer",
                "LocalOutlierFactor",
                "MiniBatchKMeans",
                "NMF",
                "OneClassSVM",
                "PCA",
            )
        ),
        "learn._src.operators.SklearnOp",
    }
)

#: Strict xfails for the 4-D contract (none open).
TIME_STACK_KNOWN_FAILURES: dict[str, Known] = {}


def _time_stack_params() -> list[Any]:
    out = []
    for cls in _CLASSES:
        key = _key(cls)
        marks = []
        if key in TIME_STACK_KNOWN_FAILURES:
            reason, raises = TIME_STACK_KNOWN_FAILURES[key]
            marks.append(pytest.mark.xfail(reason=reason, raises=raises, strict=True))
        out.append(pytest.param(cls, id=key, marks=marks))
    return out


def _stack_of(scene: Any) -> Any:
    """A 2-frame ``(T, C, H, W)`` stack whose second frame is a perturbed copy.

    Float scenes are rescaled per pixel so the frames differ; integer (label
    / QA) scenes repeat. A 2-D scene becomes ``(T, 1, H, W)``.
    """
    from _helpers import toy_geotensor

    values = np.asarray(scene)
    if values.ndim == 2:
        values = values[None]
    second = values.copy()
    if values.dtype.kind == "f":
        rng = np.random.default_rng(1)
        second = values * rng.uniform(0.8, 1.2, values.shape)
    return toy_geotensor(
        np.stack([values, second]),
        fill_value_default=scene.fill_value_default,
        attrs=dict(scene.attrs),
    )


def _build_seeded(cls: type) -> Operator:
    """``build(cls)`` with ``seed=0`` for stochastic operators."""
    op = build(cls, runtime=True)
    if "seed" in inspect.signature(cls.__init__).parameters:
        op.seed = 0
    return op


def _as_frame_result(value: Any) -> Any:
    """Per-frame carrier result as ``(C, H, W)`` (2-D maps gain a band axis)."""
    arr = np.asarray(value)
    return arr[None] if arr.ndim == 2 else arr


def _is_frame_carrier(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim in (2, 3)


def _assert_matches_frames(key: str, out: Any, expected: list[Any]) -> None:
    """``out`` equals the per-frame results restacked along time."""
    if key in TIME_INVARIANT:
        for e in expected:
            np.testing.assert_array_equal(np.asarray(out), np.asarray(e))
        return
    if all(_is_frame_carrier(e) for e in expected):
        want = np.stack([_as_frame_result(e) for e in expected])
        got = np.asarray(out)
        assert got.shape == want.shape, (
            f"4-D output shape {got.shape}; per-frame results restack to {want.shape}"
        )
        if key in STOCHASTIC_PER_PIXEL or key in POOLED_STATISTICS:
            return
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64), equal_nan=True
        )
        return
    if isinstance(expected[0], list):
        # Fan-outs (tiles): each 4-D piece restacks the per-frame pieces.
        assert isinstance(out, list) and len(out) == len(expected[0])
        for t, piece in enumerate(out):
            _assert_matches_frames(key, piece, [e[t] for e in expected])
        return
    if isinstance(expected[0], np.ndarray):
        # Stack-level statistics (a mean spectrum, a covariance) either pool
        # the frames -- same shape as one frame's statistic -- or keep one
        # statistic per frame, which must then equal the per-frame results.
        if np.shape(out) == expected[0].shape:
            return
        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float64),
            np.stack(expected).astype(np.float64),
            equal_nan=True,
        )


@pytest.mark.parametrize("cls", _time_stack_params())
def test_time_stack_contract(cls: type) -> None:
    """Every operator handles a ``(T, C, H, W)`` stack correctly or says no (#147).

    Runs each buildable single-input operator on 4-D stacks of the toy
    scenes it accepts. The output must equal the per-frame results
    restacked along time (band-collapsing results keep a singleton band
    axis, ``(T, 1, H, W)``), or the operator must reject the rank with a
    ``ValueError`` / ``TypeError`` / ``GeoToolzIOError`` naming itself --
    never a library-internal error or a silently wrong shape.
    Sequence operators (composites, mosaics) given a stack must match the
    same operator on the list of its frames.
    """
    from _helpers import frames

    from geotoolz.io._src.operators import SinkOperator

    op = _build_seeded(cls)
    if isinstance(op, SinkOperator) or not _single_carrier_in(op):
        pytest.skip("sink (see test_io) or not a single-input operator")
    key = _key(cls)
    takes_sequence = _takes_sequence(op)
    ran = False
    for scene in _attributed_inputs():
        if isinstance(scene, list) != takes_sequence:
            continue
        if takes_sequence:
            stack = _stack_of(scene[0])
            try:
                expected = _build_seeded(cls)(frames(stack))
            except Exception:
                continue
        else:
            try:
                _call(_build_seeded(cls), scene)
            except Exception:
                continue
            stack = _stack_of(scene)
            expected = None
        ran = True
        try:
            # A fresh operator per stack: stochastic operators then make the
            # same draws as the fresh per-frame references.
            out = _call(_build_seeded(cls), stack)
        except Exception as exc:
            assert_clear_rank_error(cls.__name__, exc)
            continue
        if takes_sequence:
            np.testing.assert_allclose(
                np.asarray(out, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                equal_nan=True,
            )
            continue
        if key in STACK_AS_FEATURES:
            assert np.shape(out)[-2:] == np.shape(stack)[-2:]
            continue
        per_frame = [_call(_build_seeded(cls), frame) for frame in frames(stack)]
        _assert_matches_frames(key, out, per_frame)
    if not ran:
        pytest.skip("no toy input is valid for this operator")
