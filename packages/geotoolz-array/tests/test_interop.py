"""Backends and interop: GC1 (NumPy / JAX parity, C1) and GC2 (georeader)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from affine import Affine
from conftest import CRS, TRANSFORM

from geoarray import (
    GeoArray,
    GeoArrayTraceError,
    HostTransferWarning,
    PixelWindow,
    combine,
    concatenate,
    crop,
    get_options,
    pad,
    pixel_coords,
    read_window,
    set_options,
    stack,
    write_window,
)


def _scene() -> GeoArray:
    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 1.0, (3, 12, 10))
    data[0, 1, 1] = np.nan
    data[2, 5, 7] = -1.0
    return GeoArray(
        data, TRANSFORM, CRS, fill_value=-1.0, attrs={"band_names": ["a", "b", "c"]}
    )


_WINDOW = PixelWindow(col_off=-2, row_off=8, width=5, height=6)

#: Every geoarray function that runs on both backends, as a function of a
#: carrier. GC1: the JAX result equals the NumPy one.
EXACT: dict[str, Callable[[GeoArray], Any]] = {
    "slice": lambda g: g[1:, 2:9:2, ::3],
    "band index": lambda g: g[[2, 0]],
    "isel": lambda g: g.isel(band=1, x=slice(3, None)),
    "squeeze": lambda g: g[:1].squeeze(),
    "expand_dims": lambda g: g.expand_dims(0),
    "map": lambda g: g.map(lambda d: d * 2),
    "shift_origin": lambda g: g.shift_origin((7.5, -2.5)),
    "valid_mask": lambda g: g.valid_mask(),
    "crop": lambda g: crop(g, (3, 9), (4, 4)),
    "read_window": lambda g: read_window(g, _WINDOW),
    "write_window": lambda g: write_window(g, np.zeros((3, 6, 5)), _WINDOW),
    "pad": lambda g: pad(g, ((1, 2), (0, 3))),
    "combine": lambda g: combine(lambda a, b: a - b, g, g),
    "stack": lambda g: stack([g, g]),
    "concatenate": lambda g: concatenate([g, g[:1]]),
}


def _host(result: Any) -> Any:
    return result.to_numpy() if isinstance(result, GeoArray) else np.asarray(result)


class TestBackendParity:
    """GC1: every geoarray function gives the same result on both backends."""

    @pytest.mark.parametrize("name", list(EXACT))
    def test_exact(self, name: str, x64: ModuleType) -> None:
        g = _scene()
        expected = EXACT[name](g)
        got = _host(EXACT[name](g.to_jax()))
        if isinstance(expected, GeoArray):
            assert got.identical(expected), name
        else:
            np.testing.assert_array_equal(got, expected)

    def test_pixel_coords(self, x64: ModuleType) -> None:
        rotated = (8.660254037844387, 5.0, 500000.0, 5.0, -8.660254037844387, 4e6)
        g = GeoArray(np.zeros((40, 30)), rotated, CRS)
        np.testing.assert_allclose(
            np.asarray(pixel_coords(g.to_jax())), pixel_coords(g), rtol=1e-12, atol=0
        )


class TestNumpyInterop:
    """C1: eager NumPy code works on JAX carriers; traced code raises."""

    def test_eager_asarray_copies_to_the_host(self, x64: ModuleType) -> None:
        g = _scene()
        on_jax = g.to_jax()
        host = np.asarray(on_jax)
        assert isinstance(host, np.ndarray)
        np.testing.assert_array_equal(host, on_jax.to_numpy().data)
        assert np.asarray(on_jax, dtype=np.float32).dtype == np.float32
        assert np.nansum(on_jax) == pytest.approx(np.nansum(g.data))

    def test_copy_false_is_refused_for_jax(self, x64: ModuleType) -> None:
        with pytest.raises(ValueError, match="without a copy"):
            _scene().to_jax().__array__(copy=False)

    def test_numpy_data_is_not_copied(self) -> None:
        g = _scene()
        assert np.asarray(g) is g.data
        assert np.shares_memory(g.__array__(copy=False), g.data)

    @pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
    def test_numpy_code_under_a_transform_raises(
        self, transform: str, x64: ModuleType
    ) -> None:
        def numpy_loss(c: GeoArray) -> Any:
            return np.asarray(c).sum()

        g = _scene().to_jax()
        if transform == "jit":
            run = lambda: x64.jit(numpy_loss)(g)
        elif transform == "grad":
            run = lambda: x64.grad(numpy_loss)(g)
        else:
            transforms = x64.numpy.stack([g.transform, g.transform])
            batch = GeoArray(x64.numpy.stack([g.data, g.data]), transforms, CRS)
            run = lambda: x64.vmap(numpy_loss)(batch)
        with pytest.raises(GeoArrayTraceError, match=r"geoops\.functional"):
            run()

    def test_transfers_warn_only_when_asked(self, x64: ModuleType) -> None:
        on_jax = _scene().to_jax()
        assert get_options()["warn_on_transfer"] is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            np.asarray(on_jax)
            np.asarray(_scene())
        with set_options(warn_on_transfer=True):
            with pytest.warns(HostTransferWarning, match="host"):
                np.asarray(on_jax)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                np.asarray(_scene())  # NumPy data: no transfer, no warning
        assert get_options()["warn_on_transfer"] is False

    def test_set_options_without_a_with_block(self) -> None:
        set_options(warn_on_transfer=True)
        try:
            assert get_options()["warn_on_transfer"] is True
        finally:
            set_options(warn_on_transfer=False)
        before = get_options()
        set_options()
        assert get_options() == before
        with pytest.raises(TypeError, match="bool"):
            set_options(warn_on_transfer=1)  # type: ignore[arg-type]


# -- georeader (GC2) --------------------------------------------------------

#: The grids geoops' tests use (tests/_helpers.py TOY_GRIDS), as numbers.
_ROT = (
    Affine.translation(500_000.0, 4_000_000.0)
    * Affine.rotation(30.0)
    * Affine.scale(10.0, -10.0)
)
TOY_GRIDS: dict[str, tuple[tuple[float, ...], str]] = {
    "utm": ((10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0), "EPSG:32629"),
    "non_square": ((10.0, 0.0, 500_000.0, 0.0, -20.0, 4_000_000.0), "EPSG:32629"),
    "rotated": (tuple(_ROT)[:6], "EPSG:32629"),
    "sheared": ((10.0, 5.0, 500_000.0, 0.0, -10.0, 4_000_000.0), "EPSG:32629"),
    "geographic": ((1e-4, 0.0, -8.0, 0.0, -1e-4, 53.0), "EPSG:4326"),
}


def _carriers(transform: tuple[float, ...], crs: str) -> dict[str, GeoArray]:
    """The carrier kinds geoops' tests build, on one grid."""
    rng = np.random.default_rng(0)
    scene = rng.uniform(0.0, 1.0, (3, 16, 16))
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 2:5] = 1
    return {
        "scene": GeoArray(
            scene,
            transform,
            crs,
            fill_value=0.0,
            attrs={"band_names": ["b0", "b1", "b2"]},
        ),
        "float32_nan": GeoArray(
            scene.astype(np.float32),
            transform,
            crs,
            fill_value=np.nan,
            attrs={"wavelengths": [665.0, 842.0, 1610.0], "sensor": "S2A"},
        ),
        "labels": GeoArray(labels, transform, crs, fill_value=0),
        "mask": GeoArray(labels > 0, transform, crs, fill_value=False),
        "uint16_no_fill": GeoArray(
            (scene * 1e4).astype(np.uint16)[0], transform, crs, fill_value=None
        ),
        "stack": GeoArray(
            rng.uniform(0.0, 1.0, (2, 3, 8, 8)),
            transform,
            crs,
            fill_value=np.nan,
            attrs={"tags": {"AREA_OR_POINT": "Area"}},
        ),
    }


_CASES = [
    (grid, kind) for grid, (t, c) in TOY_GRIDS.items() for kind in _carriers(t, c)
]


@pytest.mark.parametrize(("grid", "kind"), _CASES)
def test_georeader_round_trip(grid: str, kind: str) -> None:
    """GC2: from_georeader(to_georeader(g)) is identical to g, sharing memory."""
    pytest.importorskip("georeader.geotensor")
    from geoarray.interop.georeader import from_georeader, to_georeader

    transform, crs = TOY_GRIDS[grid]
    g = _carriers(transform, crs)[kind]
    gt = to_georeader(g)
    assert np.shares_memory(gt.values, g.data)
    assert gt.fill_value_default is g.fill_value or (
        g.fill_value is math.nan and math.isnan(gt.fill_value_default)
    )
    back = from_georeader(gt)
    assert back.identical(g)
    assert np.shares_memory(back.data, g.data)


def test_from_georeader_reads_georeaders_own_carriers() -> None:
    geotensor = pytest.importorskip("georeader.geotensor")
    from geoarray.interop.georeader import from_georeader

    values = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    gt = geotensor.GeoTensor(
        values,
        Affine(*TRANSFORM),
        CRS,
        fill_value_default=-9999,
        attrs={"band_names": ["B11", "B12"], "wavelengths": np.array([1610.0, 2190.0])},
    )
    g = from_georeader(gt)
    assert g.grid.transform == TRANSFORM
    assert g.crs.to_epsg() == 32613
    assert g.fill_value == -9999
    assert g.attrs["band_names"] == ("B11", "B12")
    assert g.attrs["wavelengths"] == (1610.0, 2190.0)
    no_attrs = geotensor.GeoTensor(
        values[0], Affine(*TRANSFORM), None, fill_value_default=None
    )
    assert from_georeader(no_attrs).attrs == {}
    assert from_georeader(no_attrs).crs is None


def test_adapter_errors(x64: ModuleType) -> None:
    pytest.importorskip("georeader.geotensor")
    from geoarray.interop.georeader import from_georeader, to_georeader

    g = _scene()
    with pytest.raises(TypeError, match="takes a GeoTensor"):
        from_georeader(g.data)

    class Lazy:
        def load(self) -> None: ...

    with pytest.raises(TypeError, match=r"\.load\(\)"):
        from_georeader(Lazy())
    with pytest.raises(TypeError, match="takes a GeoArray"):
        to_georeader(g.data)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="to_numpy"):
        to_georeader(g.to_jax())
    batched = GeoArray(np.zeros((2, 3, 4, 4)), np.stack([np.array(TRANSFORM)] * 2), CRS)
    with pytest.raises(ValueError, match="batched"):
        to_georeader(batched)


def test_wkt_only_crs_round_trips() -> None:
    pytest.importorskip("georeader.geotensor")
    import pyproj

    from geoarray.interop.georeader import from_georeader, to_georeader

    custom = pyproj.CRS.from_proj4("+proj=aeqd +lat_0=32 +lon_0=-104 +datum=WGS84")
    g = GeoArray(np.zeros((4, 4)), TRANSFORM, custom)
    gt = to_georeader(g)
    assert gt.crs.startswith(("PROJCRS", "PROJCS"))
    assert from_georeader(gt).identical(g)
