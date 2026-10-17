"""Model states: the reserved columns, `build_state_catalog`, `load_states`, queries.

The gate (roadmap B2, "state rows"): a cycle's ensemble history, written
as one file per run and member and indexed with `build_state_catalog`,
loads back with `load_states` with no run dropped and equal to the
history to 0 — on every backend.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely.geometry

from geocatalog import GeoCatalog, GeoSlice
from geocatalog.backends import CatalogSchemaError, InMemoryGeoCatalog
from geocatalog.storage import (
    SCHEMA_VERSION_CURRENT,
    from_geoparquet,
    migrate_geoparquet,
    to_geoparquet,
)


xr = pytest.importorskip("xarray")

import geocatalog.states as gs
from geocatalog.build import build_state_catalog

from .conftest import needs_duckdb


# Shapes: R runs, N_e members, L leads, H x W cells.
R, NE, L, H, W = 3, 4, 2, 4, 5
RUNS = pd.date_range("2026-01-01", periods=R, freq="D")
STEP = pd.Timedelta("1D")
LAT = 31.5 + 0.5 * np.arange(H)  # (H,) ascending
LON = -104.0 + 0.5 * np.arange(W)  # (W,)
AOI = GeoSlice(
    bounds=(-105.0, 31.0, -101.0, 34.0),
    interval=pd.Interval(
        pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-10"), closed="both"
    ),
    resolution=(0.5, 0.5),
    crs="EPSG:4326",
)


def _history() -> np.ndarray:
    """The ensemble history a cycle produced: ``(R, N_e, L, H, W)`` float64."""
    return np.random.default_rng(0).standard_normal((R, NE, L, H, W))


def _write_forecast(
    path: Path,
    values: np.ndarray,
    run: pd.Timestamp,
    member: int,
    *,
    lat: np.ndarray = LAT,
    lon: np.ndarray = LON,
    producer: str = "model-a",
) -> Path:
    """One (run, member) file: valid times ``r + (k + 1) Δt``, CF attributes."""
    valid = run + STEP * np.arange(1, values.shape[0] + 1)
    xr.Dataset(
        {"xch4": (("time", "lat", "lon"), values)},
        coords={"time": valid, "lat": lat, "lon": lon},
        attrs={
            "forecast_reference_time": run.isoformat(),
            "realization": member,
            "kind": "forecast",
            "producer": producer,
            "parents": f"ana_{run:%Y%m%d}.nc",
        },
    ).to_netcdf(path)
    return path


def _write_analysis(path: Path, valid: pd.Timestamp) -> Path:
    xr.Dataset(
        {"xch4": (("time", "lat", "lon"), np.zeros((1, H, W)))},
        coords={"time": [valid], "lat": LAT, "lon": LON},
        attrs={"kind": "analysis", "producer": "da-cycle"},
    ).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Path], np.ndarray]:
    """``R x N_e`` forecast files from a seeded history, plus five analyses."""
    root = tmp_path_factory.mktemp("states")
    history = _history()
    paths = [
        _write_forecast(root / f"fc_{run:%Y%m%d}_m{m}.nc", history[r, m], run, m)
        for r, run in enumerate(RUNS)
        for m in range(NE)
    ]
    paths += [
        _write_analysis(root / f"ana_{day:%Y%m%d}.nc", day)
        for day in pd.date_range(RUNS[0], periods=5, freq="D")
    ]
    return paths, history


@pytest.fixture
def catalog(
    archive: tuple[list[Path], np.ndarray],
    as_backend: Callable[[InMemoryGeoCatalog], GeoCatalog],
) -> GeoCatalog:
    paths, _ = archive
    built = build_state_catalog(paths, crs="EPSG:4326")
    assert isinstance(built, InMemoryGeoCatalog)
    return as_backend(built)


def _names(cat: GeoCatalog) -> list[str]:
    return sorted(Path(row.filepath).name for row in cat.iter_rows())


# ---------------------------------------------------------------------------
# The gate: the history comes back, every run, to 0
# ---------------------------------------------------------------------------


def test_history_round_trips_exactly(
    catalog: GeoCatalog, archive: tuple[list[Path], np.ndarray]
) -> None:
    _, history = archive
    cube = gs.load_states(gs.forecasts(catalog), AOI, data_vars=["xch4"])

    assert cube["xch4"].dims == ("init_time", "member", "lead_time", "lat", "lon")
    assert dict(cube.sizes) == {
        "init_time": R,
        "member": NE,
        "lead_time": L,
        "lat": H,
        "lon": W,
    }
    # No run dropped, although runs overlap in valid time.
    assert pd.DatetimeIndex(cube["init_time"].values).equals(RUNS)
    assert cube["member"].values.tolist() == [str(m) for m in range(NE)]
    assert pd.TimedeltaIndex(cube["lead_time"].values).equals(
        pd.TimedeltaIndex([STEP * (k + 1) for k in range(L)])
    )
    values = cube["xch4"].values  # (R, N_e, L, H, W) float64
    assert values.dtype == np.float64
    assert np.array_equal(values, history)
    assert np.abs(values - history).max() == 0.0
    # valid_time = init_time + lead_time
    expected_valid = RUNS.values[:, None] + cube["lead_time"].values[None, :]
    assert np.array_equal(cube["valid_time"].values, expected_valid)


def test_overlapping_runs_are_kept_apart(
    catalog: GeoCatalog, archive: tuple[list[Path], np.ndarray]
) -> None:
    _, history = archive
    day = RUNS[0] + 2 * STEP  # run 0 at lead 2, run 1 at lead 1
    one_day = GeoSlice(
        bounds=AOI.bounds,
        interval=pd.Interval(day, day, closed="both"),
        resolution=AOI.resolution,
        crs=AOI.crs,
    )
    cube = gs.load_states(gs.forecasts(catalog), one_day)
    assert pd.DatetimeIndex(cube["init_time"].values).equals(RUNS[:2])
    values = cube["xch4"].values  # (2, N_e, L, H, W); NaN where not valid on `day`
    assert np.array_equal(values[0, :, 1], history[0, :, 1])  # run 0, lead 2
    assert np.array_equal(values[1, :, 0], history[1, :, 0])  # run 1, lead 1
    assert np.isnan(values[0, :, 0]).all()
    assert np.isnan(values[1, :, 1]).all()


def test_missing_member_is_nan(
    catalog: GeoCatalog, archive: tuple[list[Path], np.ndarray]
) -> None:
    _, history = archive
    run1 = gs.ensemble_at(catalog, run_time=RUNS[1])
    run0_m0 = gs.forecast_chain(catalog, run_time=RUNS[0], member="0")
    rows = [*run1.iter_rows(), *run0_m0.iter_rows()]
    subset = InMemoryGeoCatalog(
        gpd.GeoDataFrame(
            [
                {
                    "filepath": r.filepath,
                    "geometry": r.geometry,
                    "start_time": r.interval.left,
                    "end_time": r.interval.right,
                    **r.extras,
                }
                for r in rows
            ],
            geometry="geometry",
            crs="EPSG:4326",
        ),
        kind="xarray",
    )
    cube = gs.load_states(subset, AOI)
    assert dict(cube.sizes)["member"] == NE
    values = cube["xch4"].values  # (2, N_e, L, H, W)
    assert np.array_equal(values[0, 0], history[0, 0])
    assert np.isnan(values[0, 1:]).all()
    assert np.array_equal(values[1], history[1])


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def test_builder_fills_every_state_column(
    archive: tuple[list[Path], np.ndarray],
) -> None:
    paths, _ = archive
    cat = build_state_catalog(paths, crs="EPSG:4326")
    gdf = cat.gdf.set_index("filepath")
    assert len(cat) == R * NE + 5
    assert gdf["run_time"].dtype == "datetime64[us]"
    fc = gdf.loc[str(paths[NE + 2])]  # run 1, member 2
    assert fc["kind"] == "forecast"
    assert fc["run_time"] == RUNS[1]
    assert fc["member"] == "2"
    assert fc["producer"] == "model-a"
    assert fc["parents"] == [f"ana_{RUNS[1]:%Y%m%d}.nc"]
    ana = gdf.loc[str(paths[-1])]
    assert ana["kind"] == "analysis"
    assert pd.isna(ana["run_time"]) and pd.isna(ana["member"])
    assert ana["parents"] is None
    row = next(r for r in cat.iter_rows() if r.filepath == str(paths[NE + 2]))
    assert row.state == gs.StateFields(
        kind="forecast",
        run_time=RUNS[1],
        member="2",
        producer="model-a",
        parents=(f"ana_{RUNS[1]:%Y%m%d}.nc",),
    )


def test_builder_with_callable_names(tmp_path: Path) -> None:
    path = _write_forecast(
        tmp_path / "fc_20260105_m7.nc", np.ones((1, H, W)), RUNS[0], 0
    )
    names = {
        **gs.CF_NAMES,
        "run_time": lambda p: pd.Timestamp(Path(p).name[3:11]),
        "member": lambda p: Path(p).stem.split("_m")[1],
    }
    cat = build_state_catalog([path], crs="EPSG:4326", names=names)
    state = next(cat.iter_rows()).state
    assert state is not None
    assert (state.run_time, state.member) == (pd.Timestamp("2026-01-05"), "7")


def test_builder_reads_cfgrib_layout(tmp_path: Path) -> None:
    run = pd.Timestamp("2026-01-01 12:00")
    steps = pd.to_timedelta([6, 12], unit="h")
    values = np.arange(2 * H * W, dtype="float64").reshape(2, H, W)
    path = tmp_path / "grib.nc"
    xr.Dataset(
        {"ch4": (("step", "latitude", "longitude"), values)},
        coords={
            "time": run,
            "step": steps,
            "valid_time": ("step", run + steps),
            "number": 3,
            "latitude": LAT,
            "longitude": LON,
        },
        attrs={"kind": "forecast"},
    ).to_netcdf(path)

    cat = build_state_catalog([path], crs="EPSG:4326", names=gs.CFGRIB_NAMES)
    row = next(cat.iter_rows())
    assert row.state is not None
    assert (row.state.run_time, row.state.member) == (run, "3")
    # The leads are the row's valid times minus the run time: τ = step.
    assert row.interval.left - run == steps[0]
    assert row.interval.right - run == steps[1]

    cube = gs.load_states(cat, AOI)
    assert dict(cube.sizes) == {
        "init_time": 1,
        "member": 1,
        "lead_time": 2,
        "latitude": H,
        "longitude": W,
    }
    assert pd.TimedeltaIndex(cube["lead_time"].values).equals(steps)
    assert np.array_equal(cube["ch4"].values[0, 0], values)


def test_builder_rejects_an_unknown_kind(tmp_path: Path) -> None:
    path = _write_forecast(tmp_path / "x.nc", np.ones((1, H, W)), RUNS[0], 0)
    with pytest.raises(ValueError, match="kind 'hindcast'"):
        build_state_catalog(
            [path], crs="EPSG:4326", names={**gs.CF_NAMES, "kind": lambda p: "hindcast"}
        )
    with pytest.raises(ValueError, match="unknown field"):
        build_state_catalog([path], crs="EPSG:4326", names={"lead": "step"})


@needs_duckdb
def test_streamed_build_keeps_state_types(
    archive: tuple[list[Path], np.ndarray], tmp_path: Path
) -> None:
    paths, history = archive
    # Analyses first: the first batch has an all-null `member` column.
    out = tmp_path / "states.parquet"
    cat = build_state_catalog(
        paths[::-1], crs="EPSG:4326", engine="duckdb", out_path=out, batch_size=4
    )
    schema = pq.read_schema(out)
    assert str(schema.field("member").type) == "string"
    assert str(schema.field("run_time").type) == "timestamp[us]"
    assert str(schema.field("parents").type) == "list<element: string>"
    cube = gs.load_states(gs.forecasts(cat), AOI)
    assert np.array_equal(cube["xch4"].values, history)


# ---------------------------------------------------------------------------
# Typed queries, on every backend
# ---------------------------------------------------------------------------


def test_typed_queries_filter_on_state_columns(catalog: GeoCatalog) -> None:
    assert len(gs.forecasts(catalog)) == R * NE

    chain = gs.forecast_chain(catalog, run_time=RUNS[1], member="2")
    assert _names(chain) == ["fc_20260102_m2.nc"]

    run = gs.ensemble_at(catalog, run_time=RUNS[0])
    assert _names(run) == [f"fc_20260101_m{m}.nc" for m in range(NE)]
    assert sorted(r.state.member for r in run.iter_rows() if r.state) == [
        "0",
        "1",
        "2",
        "3",
    ]

    # Every file holds leads 1 and 2 days: the lead is derived per row.
    assert (
        len(gs.lead_window(catalog, pd.Timedelta("2D"), pd.Timedelta("3D"))) == R * NE
    )
    assert len(gs.lead_window(catalog, pd.Timedelta("0D"), pd.Timedelta("12h"))) == 0
    assert len(gs.lead_window(catalog, pd.Timedelta("3D"), pd.Timedelta("4D"))) == 0

    # Valid on 01-03: runs 0 and 1 (every member) and that day's analysis.
    jan3 = gs.at_time(catalog, pd.Timestamp("2026-01-03"))
    assert _names(jan3) == [
        "ana_20260103.nc",
        *(f"fc_20260101_m{m}.nc" for m in range(NE)),
        *(f"fc_20260102_m{m}.nc" for m in range(NE)),
    ]

    latest = gs.latest_analysis(catalog, before=pd.Timestamp("2026-01-03 06:00"))
    assert latest is not None
    assert Path(latest.filepath).name == "ana_20260103.nc"
    assert gs.latest_analysis(catalog, before=pd.Timestamp("2025-12-31")) is None


def test_typed_queries_refuse_untyped_arguments(catalog: GeoCatalog) -> None:
    with pytest.raises(TypeError, match="member must be a string"):
        gs.forecast_chain(catalog, run_time=RUNS[0], member=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"run_time must be a pd\.Timestamp"):
        gs.ensemble_at(catalog, run_time="2026-01-01")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"pd\.Timedelta"):
        gs.lead_window(catalog, "1D", "2D")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after end"):
        gs.lead_window(catalog, pd.Timedelta("2D"), pd.Timedelta("1D"))


def test_queries_need_state_columns(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["a.nc"],
            "start_time": [RUNS[0]],
            "end_time": [RUNS[0]],
        },
        geometry=[shapely.geometry.box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    plain = InMemoryGeoCatalog(gdf, kind="xarray")
    with pytest.raises(ValueError, match="build_state_catalog"):
        gs.forecasts(plain)
    with pytest.raises(ValueError, match="build_state_catalog"):
        gs.load_states(plain, AOI)


# ---------------------------------------------------------------------------
# Loader guards
# ---------------------------------------------------------------------------


def test_load_states_guards(
    archive: tuple[list[Path], np.ndarray], tmp_path: Path
) -> None:
    paths, _ = archive
    cat = build_state_catalog(paths, crs="EPSG:4326")
    with pytest.raises(ValueError, match="no run_time"):
        gs.load_states(cat, AOI)  # the analyses have no run
    twice = build_state_catalog([paths[0], paths[0]], crs="EPSG:4326")
    with pytest.raises(ValueError, match="appears twice"):
        gs.load_states(twice, AOI)


def test_grids_split_by_group(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    fine = rng.standard_normal((2, NE, L, H, W))
    coarse = rng.standard_normal((1, 1, L, 2, 3))
    paths = [
        _write_forecast(tmp_path / f"a_{r}_{m}.nc", fine[r, m], RUNS[r], m)
        for r in range(2)
        for m in range(NE)
    ]
    paths.append(
        _write_forecast(
            tmp_path / "b_0_0.nc",
            coarse[0, 0],
            RUNS[0],
            NE,  # a member of its own: one grid per producer
            lat=LAT[::2],
            lon=LON[::2][:3],
            producer="model-b",
        )
    )
    cat = build_state_catalog(paths, crs="EPSG:4326")
    with pytest.raises(ValueError, match="different grid"):
        gs.load_states(cat, AOI)
    tree = gs.load_states(cat, AOI, group_by="producer")
    assert isinstance(tree, xr.DataTree)
    assert sorted(tree.children) == ["model-a", "model-b"]
    leaf_a = tree["model-a"].to_dataset()
    leaf_b = tree["model-b"].to_dataset()
    assert np.array_equal(leaf_a["xch4"].values, fine)
    assert np.array_equal(leaf_b["xch4"].values, coarse)


# ---------------------------------------------------------------------------
# Schema v1: reserved columns, migration from v0
# ---------------------------------------------------------------------------


def _hand_filled_v0(tmp_path: Path) -> Path:
    """A v0 artifact whose extras were filled by hand, as before v1."""
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["fc_m0.nc", "ana.nc"],
            "start_time": [RUNS[1], RUNS[1]],
            "end_time": [RUNS[2], RUNS[1]],
            "run_time": [RUNS[0], pd.NaT],
            "member": [0, -1],
            "kind": ["forecast", "analysis"],
            "cloud": [1.5, 2.5],
        },
        geometry=[shapely.geometry.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
    )
    out = tmp_path / "v0.parquet"
    to_geoparquet(InMemoryGeoCatalog(gdf, kind="xarray"), out, schema_version=0)
    return out


def test_v0_catalog_migrates_to_v1(tmp_path: Path) -> None:
    assert SCHEMA_VERSION_CURRENT == 1
    out = _hand_filled_v0(tmp_path)
    cat = from_geoparquet(out)
    gdf = cat.gdf
    assert [c for c in gs.STATE_COLUMNS if c not in gdf.columns] == []
    assert gdf["run_time"].dtype == "datetime64[us]"
    assert gdf["member"].tolist() == ["0", "-1"]
    assert gdf["producer"].isna().all()
    assert gdf["parents"].tolist() == [None, None]
    assert gdf["cloud"].tolist() == [1.5, 2.5]  # every other column unchanged
    assert gs.forecast_chain(cat, run_time=RUNS[0], member="0").gdf[
        "filepath"
    ].tolist() == ["fc_m0.nc"]

    # `geocatalog migrate` rewrites the file at v1 (what DuckDB needs).
    assert migrate_geoparquet(out, to_version=1) == 0
    table = pq.read_table(out)
    assert set(table.column("_schema_version").to_pylist()) == {1}
    member_type = table.schema.field("member").type  # pandas writes large_string
    assert pa.types.is_string(member_type) or pa.types.is_large_string(member_type)


def test_v0_catalog_without_state_columns_is_unchanged(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"filepath": ["a.tif"], "start_time": [RUNS[0]], "end_time": [RUNS[0]]},
        geometry=[shapely.geometry.box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    out = tmp_path / "plain.parquet"
    to_geoparquet(InMemoryGeoCatalog(gdf, kind="raster"), out, schema_version=0)
    cat = from_geoparquet(out)
    assert list(cat.gdf.columns) == ["filepath", "geometry"]


@needs_duckdb
def test_duckdb_refuses_an_unmigrated_v0_state_catalog(tmp_path: Path) -> None:
    from geocatalog.backends import DuckDBGeoCatalog

    out = _hand_filled_v0(tmp_path)
    with pytest.raises(CatalogSchemaError, match="geocatalog migrate"):
        DuckDBGeoCatalog.open(out)
    migrate_geoparquet(out, to_version=SCHEMA_VERSION_CURRENT)
    with DuckDBGeoCatalog.open(out) as cat:
        assert _names(gs.forecasts(cat)) == ["fc_m0.nc"]


def test_v1_types_round_trip(
    tmp_path: Path, as_backend: Callable[[InMemoryGeoCatalog], GeoCatalog]
) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["cmip.nc", "det.nc"],
            "start_time": [RUNS[0], RUNS[0]],
            "end_time": [RUNS[1], RUNS[1]],
            "kind": ["reanalysis", "forecast"],
            "run_time": [RUNS[0], RUNS[0]],
            "member": ["r1i1p1f1", None],
            "producer": ["CESM2", None],
            "parents": [["a.nc", "b.nc"], None],
        },
        geometry=[shapely.geometry.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
    )
    from geocatalog._src._schema import coerce_state_columns

    cat = as_backend(InMemoryGeoCatalog(coerce_state_columns(gdf), kind="xarray"))
    out = tmp_path / "v1.parquet"
    to_geoparquet(InMemoryGeoCatalog(cat.gdf, kind="xarray"), out)
    back = from_geoparquet(out).gdf
    assert back["run_time"].dtype == "datetime64[us]"
    assert back["member"].tolist()[0] == "r1i1p1f1"
    assert pd.isna(back["member"].tolist()[1])
    assert back["parents"].tolist() == [["a.nc", "b.nc"], None]
    assert back["kind"].tolist() == ["reanalysis", "forecast"]
    states = {r.filepath: r.state for r in cat.iter_rows()}
    assert states["cmip.nc"] == gs.StateFields(
        kind="reanalysis",
        run_time=RUNS[0],
        member="r1i1p1f1",
        producer="CESM2",
        parents=("a.nc", "b.nc"),
    )
    assert states["det.nc"] is not None
    assert states["det.nc"].member is None
    assert states["det.nc"].parents == ()
