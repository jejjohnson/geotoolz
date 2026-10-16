"""Tests for `geopatcher.integrations.train.XarrayWindowDataset`.

The pure-logic tests (collation, indexability, hashing, splits, pickling)
need only `xarray` + `numpy` + `geopatcher`. The block-reader / adapter /
resume / sharding tests additionally need `grain`; they `importorskip` it.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


# Hard requirements for every test in this module.
pytest.importorskip("xarray")
pytest.importorskip("pipekit_train")

import xarray as xr
from pipekit_train.adapters.equinox import _is_indexable

from geopatcher.integrations.train import (
    XarrayWindowDataset,
    _split_origins,
)
from geopatcher.temporal.stencils import TimeStencil


# --- Helpers --------------------------------------------------------------


def _store(n_time: int = 100, n_x: int = 2, variables=("t2m", "u10", "v10")):
    """A small in-memory hourly store standing in for a lazy Zarr."""
    time = np.arange(
        "2020-01-01",
        np.datetime64("2020-01-01") + np.timedelta64(n_time, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    rng = np.random.RandomState(0)
    data = {v: (("time", "x"), rng.standard_normal((n_time, n_x))) for v in variables}
    return xr.Dataset(data, coords={"time": time, "x": list(range(n_x))})


def _stencils():
    # input window: 4 hourly points [-3h, 0h]; target: single point at +1h.
    return {
        "input": TimeStencil("-3h", "0h", "1h", closed="both"),
        "target": TimeStencil("1h", "1h", "0h", closed="both"),
    }


def _dataset(split="train", **kw):
    ds = _store()
    return XarrayWindowDataset(
        source={"input": ds[["t2m", "u10", "v10"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        sample_dim="time",
        store_version="era5-test.1",
        split=split,
        seed=0,
        **kw,
    )


# --- Indexability + collation --------------------------------------------


def test_is_indexable():
    """Indexable -> rides the adapter's Grain path (shuffle + resume)."""
    assert _is_indexable(_dataset())


def test_getitem_returns_collated_arrays():
    d = _dataset()
    x, y = d[0]
    # input has 3 vars over a 4-point window; target 1 var, single point.
    assert x.shape == (3, 4, 2)
    assert y.shape == (1, 1, 2)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)


def test_channel_order_is_var_order():
    """Channel k of the collated array is data var k in var_order."""
    ds = _store()
    d = XarrayWindowDataset(
        source={"input": ds[["t2m", "u10", "v10"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="v1",
    )
    assert d.var_order["input"] == ["t2m", "u10", "v10"]
    x, _ = d[0]
    # The prepared source leads with `time`; channel 0 must equal t2m's window.
    expected_t2m = (
        d.source["input"]["t2m"]
        .isel(time=d._grain_source.sources["input"].slices[0])
        .values
    )
    assert np.array_equal(x[0], expected_t2m)


def test_iter_matches_getitem():
    d = _dataset()
    first = next(iter(d))
    assert np.array_equal(first[0], d[0][0])


# --- Content hashing ------------------------------------------------------


def test_hash_stable_across_construction():
    assert _dataset().content_hash() == _dataset().content_hash()


def test_hash_changes_with_split():
    d = _dataset()
    assert d.content_hash() != d.with_split("val").content_hash()


def test_hash_changes_with_stencil():
    ds = _store()
    base = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="v1",
    )
    changed = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils={
            "input": TimeStencil("-6h", "0h", "1h", closed="both"),
            "target": TimeStencil("1h", "1h", "0h", closed="both"),
        },
        store_version="v1",
    )
    assert base.content_hash() != changed.content_hash()


def test_hash_changes_with_store_version():
    a = _dataset()
    ds = _store()
    b = XarrayWindowDataset(
        source={"input": ds[["t2m", "u10", "v10"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="era5-test.2",  # different token
    )
    assert a.content_hash() != b.content_hash()


def test_hash_sensitive_to_var_order():
    """Channel identity matters: reordering input vars changes the hash."""
    ds = _store()
    a = XarrayWindowDataset(
        source={"input": ds[["t2m", "u10"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="v1",
    )
    b = XarrayWindowDataset(
        source={"input": ds[["u10", "t2m"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="v1",
    )
    assert a.content_hash() != b.content_hash()


# --- Splits + guard band --------------------------------------------------


def test_splits_are_disjoint():
    tr = _dataset("train")
    va = tr.with_split("val")
    te = tr.with_split("test")
    s_tr = set(tr.sample_origins.tolist())
    s_va = set(va.sample_origins.tolist())
    s_te = set(te.sample_origins.tolist())
    assert s_tr.isdisjoint(s_va)
    assert s_va.isdisjoint(s_te)
    assert s_tr.isdisjoint(s_te)


def test_split_guard_band_creates_gap():
    """With a positive guard, adjacent splits are separated by > guard steps."""
    origins = np.arange(100)
    guard = 2
    tr = _split_origins(origins, "train", guard)
    va = _split_origins(origins, "val", guard)
    # train's max origin and val's min origin straddle the 80% boundary;
    # the guard band leaves a gap of more than `guard` between them.
    assert va.min() - tr.max() > guard


def test_split_guard_can_empty_a_narrow_band():
    """A guard wider than half a band collapses it to empty, not an error."""
    origins = np.arange(100)
    # val band is 10 wide; trimming guard=5 from both interior sides empties it.
    assert _split_origins(origins, "val", 5).size == 0


def test_zero_guard_keeps_contiguous_tenths():
    origins = np.arange(100)
    tr = _split_origins(origins, "train", 0)
    va = _split_origins(origins, "val", 0)
    te = _split_origins(origins, "test", 0)
    assert tr.tolist() == list(range(0, 80))
    assert va.tolist() == list(range(80, 90))
    assert te.tolist() == list(range(90, 100))


def test_auto_guard_derived_from_span():
    # input span is -3h..0h over hourly data -> 3 source steps.
    assert _dataset()._split_guard == 3


# --- Validation -----------------------------------------------------------


def test_mismatched_keys_raise():
    ds = _store()
    with pytest.raises(ValueError, match="matching keys"):
        XarrayWindowDataset(
            source={"input": ds[["t2m"]]},
            stencils={"input": _stencils()["input"], "target": _stencils()["target"]},
            store_version="v1",
        )


# --- Pickling -------------------------------------------------------------


def test_pickle_round_trip():
    """Grain pickles sources across workers; __getitem__ must survive it."""
    d = _dataset()
    x0, y0 = d[0]
    rt = pickle.loads(pickle.dumps(d))
    x1, y1 = rt[0]
    assert np.array_equal(x0, x1)
    assert np.array_equal(y0, y1)


# --- Block reader / adapter hook (need grain) -----------------------------


def test_build_batch_iter_declines_without_block_reader():
    assert _dataset().build_batch_iter(4, 0) is None


def test_adapter_uses_block_reader_and_resumes():
    pytest.importorskip("grain")
    from pipekit_train.adapters.equinox import _BatchSource

    cloud = _dataset(block_reader=True, buffer_size_in_bytes=5e3, buffer_diversity=4)
    bs = _BatchSource(cloud, batch_size=4, seed=0)
    assert bs._kind == "grain"
    x, y = bs.next_batch()
    assert np.asarray(x).shape == (4, 3, 4, 2)
    assert np.asarray(y).shape == (4, 1, 1, 2)

    state = bs.get_state()
    a = bs.next_batch()
    bs.set_state(state)
    b = bs.next_batch()
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_block_reader_shards_are_disjoint():
    pytest.importorskip("grain")

    def first_batch(shard_index):
        d = _dataset(
            block_reader=True,
            buffer_size_in_bytes=5e3,
            buffer_diversity=4,
            shard_index=shard_index,
            shard_count=2,
        )
        return np.asarray(next(d.build_batch_iter(2, 0))[0])

    assert not np.array_equal(first_batch(0), first_batch(1))


# --- Regression: strided stencils, single-point guard, object hashing -----


def test_group_slices_preserves_step():
    """Block sub-slices must keep each window's stride, else the block path
    yields every point in the span instead of the strided points."""
    from geopatcher._src.xreader import group_slices

    slices = [slice(0, 10, 3), slice(1, 11, 3)]
    (_block, subs) = group_slices(slices, 2)[0]
    assert [s.step for s in subs] == [3, 3]


def test_block_reader_matches_strided_window_length():
    """A 3-hourly window over hourly data has 4 points on BOTH read paths."""
    pytest.importorskip("grain")
    ds = _store(n_time=200)
    stencils = {
        "input": TimeStencil("-9h", "0h", "3h", closed="both"),  # 4 strided pts
        "target": TimeStencil("3h", "3h", "0h", closed="both"),
    }
    d = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils=stencils,
        store_version="v1",
        block_reader=True,
        buffer_size_in_bytes=5e3,
        buffer_diversity=4,
    )
    # per-example path: 4 strided points (not 10 contiguous)
    assert d[0][0].shape == (1, 4, 2)
    # block path must agree
    xb, _ = next(d.build_batch_iter(2, 0))
    assert np.asarray(xb).shape == (2, 1, 4, 2)


def test_guard_accounts_for_single_point_offset():
    """A +24h single-point target has zero span but reaches 24 steps; the
    guard must use reach, not span, or train targets leak into val."""
    ds = _store(n_time=600)
    stencils = {
        "input": TimeStencil("0h", "0h", "0h", closed="both"),  # point at origin
        "target": TimeStencil("24h", "24h", "0h", closed="both"),  # +24h
    }
    d = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils=stencils,
        store_version="v1",
    )
    assert d._split_guard == 24
    train_max = d.sample_origins.max()
    val_min = d.with_split("val").sample_origins.min()
    # the +24h target window of the last train origin must not reach val.
    assert val_min - train_max > np.timedelta64(24, "h")


def test_empty_split_builds_length_zero():
    """A guard band wider than a band empties it; construction must not crash."""
    ds = _store()
    d = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        store_version="v1",
        split="val",
        split_guard=999,  # far wider than the val band
    )
    assert len(d) == 0
    assert list(d) == []


def test_object_dtype_origins_hash_by_value():
    """Object-typed origins (e.g. cftime) must hash by value, not pointer."""
    import datetime

    from geopatcher._src._digest import array_digest

    a = np.array([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)], object)
    b = np.array([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)], object)
    c = np.array([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 3)], object)
    assert array_digest(a) == array_digest(b)  # equal values → equal hash
    assert array_digest(a) != array_digest(c)  # different value → different hash


# --- Construction validation, block-reader edge cases, shard discovery ----


def test_source_keys_must_be_input_and_target():
    ds = _store()
    with pytest.raises(ValueError, match="exactly the keys"):
        XarrayWindowDataset(
            source={"x": ds[["t2m"]], "y": ds[["t2m"]]},
            stencils={"x": _stencils()["input"], "y": _stencils()["target"]},
            store_version="v1",
        )


@pytest.mark.parametrize("guard", [-1, 1.5, True])
def test_split_guard_must_be_a_non_negative_integer(guard):
    with pytest.raises(ValueError, match="split_guard"):
        _dataset(split_guard=guard)


@pytest.mark.parametrize(
    ("shard_index", "shard_count"), [(2, 2), (-1, 2), (0, 0), (0, -1)]
)
def test_invalid_shards_are_rejected(shard_index, shard_count):
    with pytest.raises(ValueError, match="shard"):
        _dataset(block_reader=True, shard_index=shard_index, shard_count=shard_count)


def test_block_reader_handles_an_empty_split():
    """A guard band that empties a split gives an empty block iterator."""
    pytest.importorskip("grain")
    d = _dataset(split="val", split_guard=999, block_reader=True)
    assert len(d) == 0
    with pytest.raises(StopIteration):
        next(d.build_batch_iter(2, 0))


def test_close_releases_block_reader_pools():
    pytest.importorskip("grain")
    d = _dataset(block_reader=True, buffer_size_in_bytes=5e3, buffer_diversity=4)
    next(d.build_batch_iter(2, 0))
    [block_source] = d._block_sources
    loaders = [leaf.__dict__.get("loader") for leaf in block_source.sources.values()]
    assert all(loader is not None for loader in loaders)  # pools were created
    d.close()
    assert all(loader._executor._shutdown for loader in loaders)
    assert all("loader" not in leaf.__dict__ for leaf in block_source.sources.values())


def test_shards_fall_back_to_one_host_without_jax(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "jax", None)  # import jax raises ImportError
    assert _dataset(block_reader=True)._resolve_shards() == (0, 1)


def test_jax_process_discovery_errors_propagate(monkeypatch):
    jax = pytest.importorskip("jax")

    def broken() -> int:
        raise RuntimeError("distributed backend not initialised")

    monkeypatch.setattr(jax, "process_index", broken)
    with pytest.raises(RuntimeError, match="not initialised"):
        _dataset(block_reader=True)._resolve_shards()


def test_sample_dim_needs_an_in_memory_index():
    """Without an index the sample coord would be read from the store."""
    ds = _store().drop_indexes("time")
    with pytest.raises(ValueError, match="in-memory index"):
        XarrayWindowDataset(
            source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
            stencils=_stencils(),
            store_version="v1",
        )


def test_hash_changes_with_source_view():
    """Two views of one store version must not share a content hash."""
    ds = _store(n_x=3)

    def build(view):
        return XarrayWindowDataset(
            source={"input": view[["t2m"]], "target": view[["t2m"]]},
            stencils=_stencils(),
            store_version="era5-test.1",
        ).content_hash()

    full = build(ds)
    assert build(ds.isel(x=slice(0, 2))) != full  # spatial subset
    assert build(ds.assign_coords(x=[10, 11, 12])) != full  # other locations
    assert build(ds) == full


def test_every_shard_gets_a_block():
    """A short split with the default large buffer still feeds every host."""
    pytest.importorskip("grain")
    from geopatcher._src import xreader

    d = _dataset(split="val")
    n = len(d.sample_origins)
    assert n >= 3
    _, block_source = xreader.training_iterator(
        d.source, d.stencils, d.sample_origins, shard_index=0, shard_count=3
    )
    assert len(block_source) >= 3


def test_fewer_examples_than_shards_raise():
    pytest.importorskip("grain")
    from geopatcher._src import xreader

    d = _dataset(split="val")
    with pytest.raises(ValueError, match="cannot feed"):
        xreader.training_iterator(
            d.source,
            d.stencils,
            d.sample_origins,
            shard_index=0,
            shard_count=len(d.sample_origins) + 1,
        )


def test_group_slices_ends_a_block_at_a_gap():
    """Sparse origins must not coalesce into one read spanning the gap."""
    from geopatcher._src.xreader import group_slices

    groups = group_slices([slice(0, 4), slice(1, 5), slice(1000, 1004)], 10)
    assert [block for block, _ in groups] == [slice(0, 5), slice(1000, 1004)]
    assert groups[1][1] == [slice(0, 4, None)]


def test_block_reader_pairs_leaves_with_sparse_origins():
    """Leaves with different window widths must share block boundaries.

    Origins every 3 steps: the 4-point input stencil and 1-point target
    stencil would hit their span limits at different examples if each leaf
    were grouped on its own, mis-pairing inputs and targets.
    """
    pytest.importorskip("grain")
    from geopatcher._src import xreader

    ds = _store()
    d = XarrayWindowDataset(
        source={"input": ds[["t2m", "u10", "v10"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        sample_origins=ds.time.values[3:-1:3],
        store_version="v1",
        split_guard=0,
    )
    per_example = sorted(np.concatenate([x.ravel(), y.ravel()]).tobytes() for x, y in d)
    bytes_per_example = xreader._example_nbytes(d.source, d.stencils, "time")
    it, block_source = xreader.training_iterator(
        d.source,
        d.stencils,
        d.sample_origins,
        num_epochs=1,
        buffer_size_in_bytes=4 * bytes_per_example,  # group_size 4 by count
        buffer_diversity=1,
    )
    groups = {name: leaf.groups for name, leaf in block_source.sources.items()}
    assert [len(subs) for _, subs in groups["input"]] == [
        len(subs) for _, subs in groups["target"]
    ]
    blocked = sorted(
        np.concatenate([x.ravel(), y.ravel()]).tobytes()
        for x, y in map(d._collate_pair, it)
    )
    assert blocked == per_example


def test_leaf_with_only_static_variables_is_rejected():
    ds = _store()
    static = ds[["t2m"]].isel(time=0, drop=True).assign_coords(time=ds.time)
    with pytest.raises(ValueError, match="no data variables along"):
        XarrayWindowDataset(
            source={"input": ds[["t2m"]], "target": static},
            stencils=_stencils(),
            store_version="v1",
        )


def test_caller_origins_are_snapshotted():
    ds = _store()
    origins = ds.time.values[3:-1].copy()
    d = XarrayWindowDataset(
        source={"input": ds[["t2m"]], "target": ds[["t2m"]]},
        stencils=_stencils(),
        sample_origins=origins,
        store_version="v1",
    )
    before = (d.content_hash(), d.sample_origins.copy())
    origins[:] = origins[::-1]  # the caller reuses their buffer
    assert d.content_hash() == before[0]
    np.testing.assert_array_equal(d.sample_origins, before[1])
    assert not d.sample_origins.flags.writeable
