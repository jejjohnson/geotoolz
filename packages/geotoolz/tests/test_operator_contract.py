"""Package-wide ``pipekit.Operator`` contract tests.

Walks every ``geotoolz`` operator class and checks, without optional
extras such as hydra-zen, that:

* ``Operator.from_state(op.state)`` rebuilds an equal operator, or raises
  ``RuntimeError`` for ``forbid_in_yaml`` classes;
* ``get_config()`` is strict JSON (no NaN / inf) and names only
  constructor parameters;
* calling the operator on ``Input`` nodes builds a graph ``Node``.

Operators that need constructor arguments get them from ``CTOR_KWARGS``.
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
from _helpers import all_operator_classes
from pipekit import Input, Node, Operator


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
    "geom._src.coregister.operators.SwathToGrid": {
        "target_crs": "EPSG:4326",
        "target_res": (0.1, 0.1),
    },
    "geom._src.coregister.operators.VectorToRasterAgg": {"agg": "count"},
    "geom._src.operators.BowtieCorrection": {
        "scan_angle_max_deg": 55.0,
        "pixels_per_scan": 10,
        "scans_per_granule": 20,
    },
    "geom._src.operators.CropTo": {"shape": (4, 4)},
    "geom._src.operators.CropToBounds": {"bounds": (0.0, 0.0, 1.0, 1.0)},
    "geom._src.operators.PadTo": {"shape": (8, 8)},
    "geom._src.operators.Reproject": {"dst_crs": "EPSG:4326"},
    "geom._src.operators.Resample": {"resolution": (20.0, 20.0)},
    "geom._src.operators.Resize": {"shape": (8, 8)},
    "geom._src.operators.SlidingWindow": {"size": (4, 4)},
    "geom._src.operators.Tile": {"size": (4, 4)},
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
        f"learn.{name}": _sklearn(name)
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
    "learn.LocalOutlierFactor": lambda: {
        "estimator": __import__("sklearn.neighbors").neighbors.LocalOutlierFactor(
            novelty=True
        )
    },
    "learn.GMM": lambda: {
        "estimator": __import__("sklearn.mixture").mixture.GaussianMixture()
    },
    "learn.IPCA": lambda: {
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
    "mask._src.operators.RemoveSmallHoles": {"area_threshold": 4},
    "mask._src.operators.RemoveSmallObjects": {"min_size": 4},
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
    "qa._src.operators.MaskFromQABits": {"band_idx": 0, "bits": [3, 4]},
    "qa._src.operators.MaskFromSCL": {"band_idx": 0, "classes": [8, 9]},
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
    "radiometry._src.operators.MinMax": {"vmin": 0.0, "vmax": 1.0},
    "radiometry._src.operators.RadianceToDN": {"gain": 0.1},
    "radiometry._src.operators.RadianceToReflectance": {
        "solar_irradiance": [1900.0, 1800.0],
        "acquisition_date": _DATE,
    },
    "radiometry._src.operators.ReflectanceToRadiance": {
        "solar_irradiance": [1900.0, 1800.0],
        "acquisition_date": _DATE,
    },
    "restore._src.operators.DenoisePCA": {"n_components": 2},
    "spectral._src.operators.ApplySRF": {
        "target_center_wavelengths": [500.0, 600.0],
        "target_fwhm": [20.0, 20.0],
        "source_wavelengths": [480.0, 520.0, 580.0, 620.0],
    },
    "spectral._src.operators.BandMath": {"expression": "b0 + b1"},
    "spectral._src.operators.BandRatio": {"numerator": 0, "denominator": 1},
    "spectral._src.operators.GaussianSRF": {
        "target_center_wavelengths": [500.0, 600.0],
        "target_fwhm": [20.0, 20.0],
    },
    "spectral._src.operators.NormalizedDifference": {"a": 0, "b": 1},
    "spectral._src.operators.ReorderBands": {"order": [1, 0]},
    "spectral._src.operators.SelectBands": {"indexes": [0, 1]},
    "spectral._src.operators.SpectralBinning": {
        "target_wavelengths": [500.0, 600.0],
        "width": 20.0,
    },
    "viz._src.operators.ApplyDiscreteColormap": {
        "mapping": {0: (0.0, 0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0, 1.0)}
    },
    "viz._src.operators.Composite": {"bands": [0, 1, 2]},
    "viz._src.operators.FalseColor": {"nir": 0, "red": 1, "green": 2},
    "viz._src.operators.SWIRComposite": {"swir2": 0, "nir": 1, "red": 2},
    "viz._src.operators.TrueColor": {"red": 0, "green": 1, "blue": 2},
}

#: Operators whose constructor needs a runtime object that cannot come from a
#: config at all (another operator's fitted state, a patcher, a GeoTensor).
#: Their state round-trip is not checkable; they should be ``forbid_in_yaml``
#: (see #140) or take a config-able argument instead.
UNBUILDABLE: dict[str, str] = {
    "matched_filter._src.operators.LinearTargetFromObs": "needs an obs_model",
    "matched_filter._src.operators.NonlinearTargetFromObs": "needs an obs_model",
    "patch_ops.ApplyToChips": "needs a wrapped operator",
    "patch_ops.GridSampler": "needs a SpatialPatcher",
    "patch_ops.Stitch": "needs an aggregation and a domain",
    "plume._src.operators.CrossSectionalFlux": "needs a plume_mask GeoTensor",
    "plume._src.operators.IMEEstimate": "needs a plume_mask GeoTensor",
    "qa._src.operators._QAMask": "private base class",
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


def build(cls: type) -> Operator:
    """Construct ``cls`` from ``CTOR_KWARGS`` or with no arguments.

    Skips when the class needs runtime objects (``UNBUILDABLE``) or is a
    ``forbid_in_yaml`` class with no entry.
    """
    key = _key(cls)
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

    Keeps the contract tests below from silently skipping new classes.
    """
    missing = []
    for cls in _CLASSES:
        key = _key(cls)
        if cls.forbid_in_yaml or key in CTOR_KWARGS or key in UNBUILDABLE:
            continue
        try:
            cls()
        except (TypeError, ValueError):
            missing.append(key)
    assert missing == [], "add constructor kwargs to CTOR_KWARGS"


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
    tables = [CTOR_KWARGS, UNBUILDABLE, *KNOWN_FAILURES.values()]
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


@pytest.mark.parametrize("cls", _params("graph_mode"))
def test_graph_mode(cls: type) -> None:
    """Calling an operator on ``Input`` nodes returns a ``Node``."""
    op = build(cls)
    n = _n_inputs(op)
    if n == 0:
        pytest.skip("input-less operator (source)")
    node = op(*(Input(f"x{i}") for i in range(n)))
    assert isinstance(node, Node)
    assert node.operator is op


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
        (gz.radiometry.PercentileClip(axis=(-2, -1)), _scene),
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
