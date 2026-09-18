"""Tests for ``regrid_conservative``, ``overlap_weights_1d``, ``RegridConservative``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrcore import Signature
from xrtoolz.interpolate import (
    RegridConservative,
    coarsen_conservative,
    overlap_weights_1d,
    regrid_conservative,
)
from xrtoolz.interpolate._src.grid_to_grid import _bounds_from_centres


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _centres(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` cell centres tiling ``[lo, hi]`` (inferred bounds hit lo / hi)."""
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _cell_area(lat: np.ndarray, lon: np.ndarray, geometry: str) -> np.ndarray:
    """Cell areas ``(lat, lon)`` consistent with the inferred bounds."""
    b_lat = _bounds_from_centres(lat, "lat")
    b_lon = _bounds_from_centres(lon, "lon")
    if geometry == "spherical":
        d_lat = np.diff(np.sin(np.deg2rad(np.clip(b_lat, -90.0, 90.0))))
        d_lon = np.deg2rad(np.diff(b_lon))
    else:
        d_lat, d_lon = np.diff(b_lat), np.diff(b_lon)
    return np.outer(np.abs(d_lat), np.abs(d_lon))


def _field(lat: np.ndarray, lon: np.ndarray, seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(1.0, 2.0, size=(lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="f",
        attrs={"units": "kg m-2 s-1"},
    )


# ---------------------------------------------------------------------------
# overlap_weights_1d
# ---------------------------------------------------------------------------


def test_overlap_weights_1d_hand_computed() -> None:
    src = np.array([0.0, 1.0, 2.0, 3.0])
    # Staggered target edges.
    np.testing.assert_allclose(
        overlap_weights_1d(src, np.array([0.0, 1.5, 3.0])),
        [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]],
    )
    # Nested: target strictly inside one source cell.
    np.testing.assert_allclose(
        overlap_weights_1d(src, np.array([0.25, 0.75])), [[0.5, 0.0, 0.0]]
    )
    # Target coarser than the source.
    np.testing.assert_allclose(
        overlap_weights_1d(src, np.array([-1.0, 4.0])), [[1, 1, 1]]
    )
    # Disjoint.
    np.testing.assert_array_equal(
        overlap_weights_1d(src, np.array([10.0, 11.0])), [[0, 0, 0]]
    )
    # Descending edges: intervals keep their own order (source 0 is [2, 3],
    # target 0 is [1.5, 3]).
    np.testing.assert_allclose(
        overlap_weights_1d(src[::-1], np.array([3.0, 1.5, 0.0])),
        [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]],
    )
    assert (overlap_weights_1d(src, np.array([0.5, 2.5])) >= 0).all()


def test_overlap_weights_1d_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="at least two edges"):
        overlap_weights_1d(np.array([0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="1-D"):
        overlap_weights_1d(np.zeros((2, 2)), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# conservation / constant / identity properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["spherical", "planar"])
@pytest.mark.parametrize("n_target", [(5, 4), (30, 25)], ids=["coarser", "finer"])
def test_conservation_mean_and_sum(geometry, n_target) -> None:
    """Area integral (mean) and total (sum) survive when the target covers the
    source."""
    lat, lon = _centres(0.0, 10.0, 20), _centres(100.0, 120.0, 20)
    da = _field(lat, lon)
    tgt = {
        "lat": _centres(0.0, 10.0, n_target[0]),
        "lon": _centres(100.0, 120.0, n_target[1]),
    }
    src_area = _cell_area(lat, lon, geometry)
    tgt_area = _cell_area(tgt["lat"], tgt["lon"], geometry)

    mean = regrid_conservative(da, tgt, geometry=geometry, mode="mean")
    assert mean.dims == ("lat", "lon") and mean.shape == n_target
    assert mean.attrs == da.attrs
    np.testing.assert_allclose(
        (mean.values * tgt_area).sum(), (da.values * src_area).sum(), rtol=1e-12
    )

    total = regrid_conservative(da, tgt, geometry=geometry, mode="sum")
    np.testing.assert_allclose(total.values.sum(), da.values.sum(), rtol=1e-12)

    # Linearity: regridding a*f + b*g equals the combination of the results.
    g = _field(lat, lon, seed=1)
    lhs = regrid_conservative(2.0 * da - 0.5 * g, tgt, geometry=geometry)
    rhs = 2.0 * mean - 0.5 * regrid_conservative(g, tgt, geometry=geometry)
    np.testing.assert_allclose(lhs.values, rhs.values, rtol=1e-12)


@pytest.mark.parametrize("geometry", ["spherical", "planar"])
def test_constant_preserved_on_every_cell_including_edges(geometry) -> None:
    lat, lon = _centres(0.0, 10.0, 10), _centres(0.0, 10.0, 10)
    da = xr.full_like(_field(lat, lon), 3.5)
    # Target cells of width 2 with edges -3, -1, 1, ..., 13: the outermost
    # ring lies outside the source, the next ring is half covered.
    tgt = {"lat": _centres(-3.0, 13.0, 8), "lon": _centres(-3.0, 13.0, 8)}
    out = regrid_conservative(da, tgt, geometry=geometry)
    inner = out.isel(lat=slice(1, -1), lon=slice(1, -1))
    np.testing.assert_allclose(inner.values, 3.5, rtol=1e-12)
    assert np.isnan(out.isel(lat=0)).all() and np.isnan(out.isel(lon=-1)).all()
    # destarea scales a half-covered edge cell down by its covered fraction.
    dest = regrid_conservative(da, tgt, geometry=geometry, normalize="destarea")
    np.testing.assert_allclose(dest.isel(lat=4, lon=1).values, 3.5 * 0.5, rtol=1e-12)
    np.testing.assert_allclose(dest.isel(lat=4, lon=4).values, 3.5, rtol=1e-12)


def test_identity_target_returns_input() -> None:
    lat, lon = np.linspace(-60.0, 60.0, 13), np.linspace(-170.0, 170.0, 18)
    da = _field(lat, lon)
    for target in (da, da.to_dataset(), {"lat": lat, "lon": lon}):
        out = regrid_conservative(da, target)
        np.testing.assert_allclose(out.values, da.values, rtol=1e-12)
        np.testing.assert_array_equal(out["lat"].values, lat)
        np.testing.assert_array_equal(out["lon"].values, lon)


def test_matches_coarsen_conservative_for_aligned_integer_factor() -> None:
    """Uniform latitude spacing makes ``sin`` differences proportional to
    ``cos(lat)`` at the centres, so the two weightings agree to round-off."""
    lat = np.linspace(-87.5, 87.5, 36)
    lon = np.linspace(2.5, 357.5, 72)
    da = _field(lat, lon)
    da.values[3:5, 10:12] = np.nan  # exercise the renormalisation too
    factor = {"lat": 4, "lon": 6}
    coarse = coarsen_conservative(da, factor)
    out = regrid_conservative(da, coarse)
    np.testing.assert_allclose(out.values, coarse.values, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_handling_fracarea_destarea_and_skipna() -> None:
    lat, lon = _centres(0.0, 4.0, 4), _centres(0.0, 4.0, 4)
    da = xr.full_like(_field(lat, lon), 2.0)
    da.values[0, 0] = np.nan  # one cell of the first 2x2 block
    da.values[2:, 2:] = np.nan  # the whole last block
    tgt = {"lat": _centres(0.0, 4.0, 2), "lon": _centres(0.0, 4.0, 2)}

    frac = regrid_conservative(da, tgt, geometry="planar")
    np.testing.assert_allclose(frac.values[0, 0], 2.0)
    np.testing.assert_allclose(frac.values[0, 1], 2.0)
    assert np.isnan(frac.values[1, 1])

    dest = regrid_conservative(da, tgt, geometry="planar", normalize="destarea")
    np.testing.assert_allclose(dest.values[0, 0], 2.0 * 0.75)
    np.testing.assert_allclose(dest.values[0, 1], 2.0)
    assert np.isnan(dest.values[1, 1])

    strict = regrid_conservative(da, tgt, geometry="planar", skipna=False)
    assert np.isnan(strict.values[0, 0]) and np.isnan(strict.values[1, 1])
    np.testing.assert_allclose(strict.values[0, 1], 2.0)

    total = regrid_conservative(da, tgt, geometry="planar", mode="sum")
    np.testing.assert_allclose(total.values[0, 0], 3 * 2.0)
    assert np.isnan(total.values[1, 1])


# ---------------------------------------------------------------------------
# longitude wrap
# ---------------------------------------------------------------------------


def test_longitude_wrap_conserves_integral_and_matches_rolled_grid() -> None:
    lat = _centres(-40.0, 40.0, 8)
    lon_360 = _centres(0.0, 360.0, 36)  # bounds 0..360
    da = _field(lat, lon_360)
    src_area = _cell_area(lat, lon_360, "spherical")

    # Same cells relabelled on -180..180: the result is the rolled input.
    lon_180 = np.where(lon_360 >= 180.0, lon_360 - 360.0, lon_360)
    lon_180 = np.sort(lon_180)
    same = regrid_conservative(da, {"lat": lat, "lon": lon_180})
    expected = da.assign_coords(
        lon=np.where(lon_360 >= 180.0, lon_360 - 360.0, lon_360)
    )
    expected = expected.sortby("lon")
    np.testing.assert_allclose(same.values, expected.values, rtol=1e-12)

    # 20-degree target cells offset by 5 degrees so one straddles the seam.
    lon_tgt = np.arange(-165.0, 180.0, 20.0)  # bounds -175 .. 185
    out = regrid_conservative(da, {"lat": lat, "lon": lon_tgt})
    tgt_area = _cell_area(lat, lon_tgt, "spherical")
    np.testing.assert_allclose(
        (out.values * tgt_area).sum(), (da.values * src_area).sum(), rtol=1e-12
    )
    # Constant field stays constant through the seam cell.
    const = regrid_conservative(xr.ones_like(da), {"lat": lat, "lon": lon_tgt})
    np.testing.assert_allclose(const.values, 1.0, rtol=1e-12)
    # Planar geometry is not periodic: a -180..180 target sees only 0..180.
    planar = regrid_conservative(da, {"lat": lat, "lon": lon_tgt}, geometry="planar")
    assert np.isnan(planar.isel(lon=0)).all() and np.isfinite(planar.isel(lon=-1)).all()


def test_longitude_span_over_one_period_raises() -> None:
    lat = _centres(0.0, 10.0, 2)
    da = _field(lat, _centres(0.0, 400.0, 4))
    with pytest.raises(ValueError, match="more than one period"):
        regrid_conservative(da, {"lat": lat, "lon": [0.0, 10.0]})


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_cf_bounds_attribute_and_explicit_bounds() -> None:
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.5, 1.5])
    values = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    lat_bnds = np.array([[0.0, 0.5], [0.5, 1.5], [1.5, 3.0]])  # not midpoints
    ds = xr.Dataset(
        {"f": (("lat", "lon"), values), "lat_bnds": (("lat", "nv"), lat_bnds)},
        coords={"lat": ("lat", lat, {"bounds": "lat_bnds"}), "lon": lon},
    )
    tgt = {"lat": [1.5], "lon": [1.0]}
    tgt_bounds = {"lat": np.array([0.0, 3.0]), "lon": np.array([0.0, 2.0])}

    out = regrid_conservative(ds, tgt, geometry="planar", target_bounds=tgt_bounds)
    assert "lat_bnds" not in out  # source-grid bounds are dropped
    np.testing.assert_allclose(out["f"].values, [[(0.5 * 1 + 1.0 * 2 + 1.5 * 4) / 3.0]])

    # Explicit bounds override the CF attribute; CF-style (n, 2) accepted.
    mid = regrid_conservative(
        ds,
        tgt,
        geometry="planar",
        bounds={"lat": np.array([[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])},
        target_bounds={"lat": np.array([-0.5, 2.5]), "lon": tgt_bounds["lon"]},
    )
    np.testing.assert_allclose(mid["f"].values, [[(1 + 2 + 4) / 3.0]])

    with pytest.raises(ValueError, match="shape"):
        regrid_conservative(ds, tgt, geometry="planar", bounds={"lat": np.zeros(7)})
    with pytest.raises(ValueError, match="single centre"):
        regrid_conservative(ds, tgt, geometry="planar")


def test_gapped_cf_bounds_target_covers_only_its_cells() -> None:
    """Non-contiguous CF bounds: every other source row is a target cell; the
    gap rows contribute nothing and the covered total is conserved."""
    lat, lon = _centres(0.0, 8.0, 8), _centres(0.0, 4.0, 4)
    da = _field(lat, lon)
    tgt_lat = np.array([0.5, 2.5, 4.5, 6.5])
    lat_bnds = np.stack([tgt_lat - 0.5, tgt_lat + 0.5], axis=1)  # gaps between
    target = xr.Dataset(
        {"lat_bnds": (("lat", "nv"), lat_bnds)},
        coords={"lat": ("lat", tgt_lat, {"bounds": "lat_bnds"}), "lon": lon},
    )
    covered = da.isel(lat=[0, 2, 4, 6])

    total = regrid_conservative(da, target, geometry="planar", mode="sum")
    np.testing.assert_allclose(total.values, covered.values, rtol=1e-12)
    np.testing.assert_allclose(float(total.sum()), float(covered.sum()), rtol=1e-12)
    mean = regrid_conservative(da, target, geometry="planar")
    np.testing.assert_allclose(mean.values, covered.values, rtol=1e-12)
    # Source-side gaps: only the overlap with each cell's own interval counts.
    src_gapped = regrid_conservative(
        da,
        {"lat": [4.0], "lon": lon},
        geometry="planar",
        bounds={"lat": np.stack([lat - 0.25, lat + 0.25], axis=1)},
        target_bounds={"lat": np.array([0.0, 8.0])},
    )
    np.testing.assert_allclose(
        src_gapped.values, da.mean("lat").values[None], rtol=1e-12
    )
    # Overlap weights of gapped intervals, spelled out.
    from xrtoolz.interpolate._src.grid_to_grid import _overlap_intervals

    w = _overlap_intervals(
        np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[0.5, 2.5], [5.0, 6.0]])
    ).toarray()
    np.testing.assert_allclose(w, [[0.5, 0.5], [0.0, 0.0]])


def test_validation_errors() -> None:
    lat, lon = _centres(0.0, 4.0, 4), _centres(0.0, 4.0, 4)
    da = _field(lat, lon)
    tgt = {"lat": lat, "lon": lon}
    with pytest.raises(ValueError, match="geometry"):
        regrid_conservative(da, tgt, geometry="cubed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        regrid_conservative(da, tgt, mode="max")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalize"):
        regrid_conservative(da, tgt, normalize="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly two"):
        regrid_conservative(da, tgt, dims=("lat",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing dims"):
        regrid_conservative(da, tgt, dims=("y", "x"))
    with pytest.raises(ValueError, match="target mapping is missing"):
        regrid_conservative(da, {"lat": lat})
    with pytest.raises(ValueError, match="target is missing a coordinate"):
        regrid_conservative(da, xr.Dataset(coords={"lat": lat}))
    with pytest.raises(ValueError, match="strictly monotone"):
        regrid_conservative(da, {"lat": [0.0, 2.0, 1.0], "lon": lon})


def test_target_bounds_variables_attached_to_dataset_dropped_for_dataarray() -> None:
    lat, lon = _centres(0.0, 4.0, 8), _centres(0.0, 4.0, 4)
    da = _field(lat, lon)
    tgt_lat = np.array([1.0, 3.0])
    lat_bnds = np.array([[0.0, 2.0], [2.0, 4.0]])
    target = xr.Dataset(
        {"lat_bnds": (("lat", "nv"), lat_bnds)},
        coords={
            "lat": ("lat", tgt_lat, {"bounds": "lat_bnds", "units": "degrees"}),
            "lon": lon,
        },
    )
    ds_out = regrid_conservative(da.to_dataset(), target, geometry="planar")
    assert ds_out["lat"].attrs == {"bounds": "lat_bnds", "units": "degrees"}
    np.testing.assert_array_equal(ds_out["lat_bnds"].values, lat_bnds)
    assert ds_out["lat_bnds"].dims == ("lat", "nv")
    # Explicit target_bounds override the CF variable, so it is not attached.
    over = regrid_conservative(
        da.to_dataset(),
        target,
        geometry="planar",
        target_bounds={"lat": np.array([0.5, 1.5, 3.5])},
    )
    assert "lat_bnds" not in over and over["lat"].attrs == {"units": "degrees"}
    # DataArray output has no room for a bounds variable: no dangling attr.
    da_out = regrid_conservative(da, target, geometry="planar")
    assert da_out["lat"].attrs == {"units": "degrees"}
    assert "lat_bnds" not in da_out.coords


def test_complex_field_regrids_real_and_imaginary_parts() -> None:
    lat, lon = _centres(0.0, 6.0, 6), _centres(0.0, 6.0, 6)
    re, im = _field(lat, lon, seed=5), _field(lat, lon, seed=6)
    re.values[1, 1] = np.nan
    im.values[4, 2] = np.nan  # a cell invalid in either part is invalid
    z = re + 1j * im
    assert np.iscomplexobj(z.values)
    tgt = {"lat": _centres(0.0, 6.0, 3), "lon": _centres(0.0, 6.0, 2)}
    for kwargs in ({}, {"mode": "sum"}, {"normalize": "destarea"}, {"skipna": False}):
        out = regrid_conservative(z, tgt, geometry="planar", **kwargs)
        assert out.dtype == np.complex128
        valid = np.isfinite(re) & np.isfinite(im)
        exp_re = regrid_conservative(re.where(valid), tgt, geometry="planar", **kwargs)
        exp_im = regrid_conservative(im.where(valid), tgt, geometry="planar", **kwargs)
        np.testing.assert_allclose(out.real.values, exp_re.values, rtol=1e-12)
        np.testing.assert_allclose(out.imag.values, exp_im.values, rtol=1e-12)


# ---------------------------------------------------------------------------
# Dataset handling, extra dims, dask
# ---------------------------------------------------------------------------


def test_dataset_passthrough_and_single_dim_variable() -> None:
    lat, lon = _centres(0.0, 4.0, 4), _centres(0.0, 4.0, 4)
    f = _field(lat, lon)
    ds = xr.Dataset(
        {"f": f, "scalar": ((), 1.0), "series": (("time",), np.arange(3.0))},
        coords={"time": np.arange(3), "mask": (("lat", "lon"), np.ones((4, 4)))},
    )
    ds.attrs["title"] = "t"
    tgt = {"lat": _centres(0.0, 4.0, 2), "lon": _centres(0.0, 4.0, 2)}
    out = regrid_conservative(ds, tgt, geometry="planar")
    assert out.attrs == {"title": "t"}
    assert out["f"].shape == (2, 2)
    xr.testing.assert_identical(out["series"], ds["series"])
    assert float(out["scalar"]) == 1.0
    assert "mask" not in out.coords and "time" in out.coords

    ds["half"] = (("lat",), np.arange(4.0))
    with pytest.raises(ValueError, match="only 'lat'"):
        regrid_conservative(ds, tgt, geometry="planar")


def test_extra_dims_broadcast_and_dim_order_kept() -> None:
    lat, lon = _centres(0.0, 6.0, 6), _centres(0.0, 6.0, 6)
    rng = np.random.default_rng(3)
    da = xr.DataArray(
        rng.standard_normal((6, 3, 6)),
        dims=("lat", "time", "lon"),
        coords={"lat": lat, "lon": lon, "time": np.arange(3)},
    )
    tgt = {"lat": _centres(0.0, 6.0, 2), "lon": _centres(0.0, 6.0, 3)}
    out = regrid_conservative(da, tgt, geometry="planar")
    assert out.dims == ("lat", "time", "lon") and out.shape == (2, 3, 3)
    for t in range(3):
        per_slice = regrid_conservative(da.isel(time=t), tgt, geometry="planar")
        np.testing.assert_allclose(out.isel(time=t).values, per_slice.values)


@pytest.mark.dask
def test_dask_chunked_along_time_stays_lazy() -> None:
    pytest.importorskip("dask.array")
    lat, lon = _centres(-10.0, 10.0, 8), _centres(0.0, 20.0, 8)
    rng = np.random.default_rng(4)
    da = xr.DataArray(
        rng.standard_normal((4, 8, 8)),
        dims=("time", "lat", "lon"),
        coords={"lat": lat, "lon": lon, "time": np.arange(4)},
    )
    tgt = {"lat": _centres(-10.0, 10.0, 4), "lon": _centres(0.0, 20.0, 5)}
    eager = regrid_conservative(da, tgt)
    lazy = regrid_conservative(da.chunk({"time": 1}), tgt)
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.compute().values, eager.values)
    with pytest.raises(ValueError, match="regrid_conservative requires"):
        regrid_conservative(da.chunk({"lat": 4}), tgt)


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------


def test_operator_matches_function_and_config_round_trips() -> None:
    lat, lon = _centres(0.0, 10.0, 10), _centres(0.0, 10.0, 10)
    ds = _field(lat, lon).to_dataset()
    tgt = {"lat": _centres(0.0, 10.0, 5), "lon": _centres(0.0, 10.0, 2)}
    op = RegridConservative(tgt, mode="sum", geometry="planar")
    xr.testing.assert_allclose(
        op(ds), regrid_conservative(ds, tgt, mode="sum", geometry="planar")
    )
    cfg = op.get_config()
    assert cfg == {
        "target": {"lat": tgt["lat"].tolist(), "lon": tgt["lon"].tolist()},
        "dims": ["lat", "lon"],
        "geometry": "planar",
        "target_bounds": {
            "lat": [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]],
            "lon": [[0.0, 5.0], [5.0, 10.0]],
        },
        "mode": "sum",
        "normalize": "fracarea",
        "skipna": True,
    }
    assert json.loads(json.dumps(cfg)) == cfg
    rebuilt = RegridConservative(
        cfg["target"], **{k: v for k, v in cfg.items() if k != "target"}
    )
    assert rebuilt.get_config() == cfg
    xr.testing.assert_allclose(rebuilt(ds), op(ds))

    sig = op.compute_output_signature(
        Signature({"time": 3, "lat": 10, "lon": 10}, dtype=np.dtype("float32"))
    )
    assert sig.dims == {"time": 3, "lat": 5, "lon": 2} and sig.dtype == "float64"
    csig = op.compute_output_signature(Signature({"lat": 10, "lon": 10}, "complex64"))
    assert csig.dtype == "complex128"
    assert op.compute_output_signature(Signature({"lat": 10})).dtype == "float64"
    with pytest.raises(ValueError, match="missing"):
        RegridConservative({"lat": [0.0, 1.0]})


def test_operator_config_round_trip_keeps_irregular_cf_bounds() -> None:
    lat, lon = _centres(0.0, 3.0, 12), _centres(0.0, 2.0, 8)
    ds = _field(lat, lon).to_dataset()
    lat_bnds = np.array([[0.0, 0.5], [0.5, 1.5], [1.5, 3.0]])  # not midpoints
    target = xr.Dataset(
        {"lat_bnds": (("lat", "nv"), lat_bnds)},
        coords={
            "lat": ("lat", np.array([0.25, 1.0, 2.25]), {"bounds": "lat_bnds"}),
            "lon": np.array([0.5, 1.5]),
        },
    )
    op = RegridConservative(target, geometry="planar")
    cfg = op.get_config()
    assert cfg["target_bounds"]["lat"] == lat_bnds.tolist()
    assert json.loads(json.dumps(cfg)) == cfg
    rebuilt = RegridConservative(
        cfg["target"], **{k: v for k, v in cfg.items() if k != "target"}
    )
    expected = op(ds)
    xr.testing.assert_identical(rebuilt(ds)["f"], expected["f"])
    # ... and it genuinely differs from the midpoint inference.
    midpoints = RegridConservative(cfg["target"], geometry="planar")(ds)
    assert not np.allclose(midpoints["f"].values, expected["f"].values)
    # Explicit target_bounds on the operator win over the CF variable.
    explicit = RegridConservative(
        target, geometry="planar", target_bounds={"lat": [-0.5, 0.75, 1.5, 2.75]}
    )
    assert explicit.get_config()["target_bounds"]["lat"] == [
        [-0.5, 0.75],
        [0.75, 1.5],
        [1.5, 2.75],
    ]


def test_operator_maps_over_a_two_leaf_datatree() -> None:
    lat, lon = _centres(0.0, 10.0, 10), _centres(0.0, 10.0, 10)
    ds = _field(lat, lon).to_dataset()
    tree = xr.DataTree.from_dict({"coarse": ds, "fine": ds * 2.0})
    op = RegridConservative(ds.isel(lat=slice(0, 10, 2), lon=slice(0, 10, 5)))
    out = op(tree)
    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"coarse", "fine"}
    assert out["coarse"].dataset["f"].shape == (5, 2)
    np.testing.assert_allclose(
        out["fine"].dataset["f"].values, 2.0 * out["coarse"].dataset["f"].values
    )
