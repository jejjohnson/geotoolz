"""Feature layouts — the contract between labeled xarray data and sklearn's 2-D ``X``.

sklearn validates only the *number* of features it is given, never their
meaning. A ``(time, lat, lon)`` cube flattened in ``(lon, lat)`` order, or
cut from a different region with the same shape, reaches the estimator as
a matrix of the right width and is silently mis-scored. The layouts in
this module close that gap:

1. At fit time, :func:`array_layout` / :func:`dataset_layout` record the
   feature grid — feature dims in order, their sizes, and every coordinate
   that lives on them (index *and* auxiliary coords, with attrs).
2. At transform time, :func:`conform_array` / :func:`conform_dataset`
   check a new input against that record, reorder permuted dims and
   coordinates to the fit-time order, and raise on anything that cannot
   be matched.
3. :func:`to_2d` flattens a conformed input to ``(n_samples, n_features)``
   by a plain C-order reshape of ``(sample_dim, *feature_dims)``, and
   :func:`grid_from_2d` inverts it — so the flattening is fully determined
   by the layout, never by the order a caller happened to pass dims in.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class ArrayLayout:
    """Fit-time feature grid of one :class:`xr.DataArray`.

    Attributes:
        sample_dim: Dimension indexing samples (rows of ``X``).
        dims: The array's dimension order when the layout was recorded.
        feature_dims: Non-sample dims, in ``dims`` order. Their C-order
            flattening defines the feature columns.
        feature_shape: Sizes of ``feature_dims``.
        feature_coords: Every coordinate whose dims are a non-empty subset
            of ``feature_dims`` (index and auxiliary coords, attrs kept).
        name: The array's name.
        attrs: The array's attrs.
    """

    sample_dim: Hashable
    dims: tuple[Hashable, ...]
    feature_dims: tuple[Hashable, ...]
    feature_shape: tuple[int, ...]
    feature_coords: xr.Coordinates
    name: Hashable | None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Number of feature columns this array contributes (1 for 1-D)."""
        return math.prod(self.feature_shape)


@dataclass(frozen=True)
class DatasetLayout:
    """Fit-time feature grid of an :class:`xr.Dataset`.

    Data variables are column-concatenated in ``variables`` order;
    variable ``variables[i]`` owns columns ``bounds[i]:bounds[i + 1]``.

    Attributes:
        sample_dim: Dimension indexing samples, shared by every variable.
        variables: Data-variable names, in column order.
        arrays: One :class:`ArrayLayout` per variable.
        bounds: Column offsets, ``len(variables) + 1`` long.
        attrs: The Dataset's attrs.
    """

    sample_dim: Hashable
    variables: tuple[Hashable, ...]
    arrays: tuple[ArrayLayout, ...]
    bounds: tuple[int, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Total number of feature columns across all variables."""
        return self.bounds[-1]


@dataclass(frozen=True)
class TreeLayout:
    """Fit-time feature grid of a set of DataTree nodes joined column-wise.

    Used by ``tree_mode="concat_features"``: every node shares the sample
    axis and contributes its data variables as extra feature columns, in
    ``paths`` order; node ``paths[i]`` owns columns
    ``bounds[i]:bounds[i + 1]``.

    Attributes:
        sample_dim: Dimension indexing samples, shared by every node.
        paths: Node paths, in column order.
        datasets: One :class:`DatasetLayout` per node.
        bounds: Column offsets, ``len(paths) + 1`` long.
    """

    sample_dim: Hashable
    paths: tuple[str, ...]
    datasets: tuple[DatasetLayout, ...]
    bounds: tuple[int, ...]

    @property
    def n_features(self) -> int:
        """Total number of feature columns across all nodes."""
        return self.bounds[-1]


Layout = ArrayLayout | DatasetLayout | TreeLayout


def _coords_on(obj: xr.DataArray, dims: set[Hashable]) -> xr.Coordinates:
    """Coordinates of ``obj`` whose dims are a non-empty subset of ``dims``."""
    drop = [
        name
        for name, coord in obj.coords.items()
        if not coord.dims or not set(coord.dims) <= dims
    ]
    # Detach from ``obj`` via a coords-only Dataset so a stored layout never
    # keeps the (possibly large) training array alive.
    return obj.drop_vars(drop).coords.to_dataset().coords


def sample_coords(
    obj: xr.DataArray | xr.Dataset, sample_dim: Hashable
) -> xr.Coordinates:
    """Coordinates that label the sample axis of ``obj``.

    Returns the coords that live on ``sample_dim`` alone plus scalar
    coords — the ones that stay meaningful on any output that keeps the
    sample axis, whatever happens to the feature axis.

    Args:
        obj: The input whose sample labels should be carried to the output.
        sample_dim: The sample dimension.

    Returns:
        The matching coordinates (index coords keep their indexes).
    """
    drop = [
        name
        for name, coord in obj.coords.items()
        if coord.dims and tuple(coord.dims) != (sample_dim,)
    ]
    return obj.drop_vars(drop).coords.to_dataset().coords


def array_layout(da: xr.DataArray, sample_dim: Hashable) -> ArrayLayout:
    """Record the feature grid of ``da``.

    Args:
        da: The array to describe.
        sample_dim: The dimension indexing samples.

    Returns:
        The array's :class:`ArrayLayout`.

    Raises:
        ValueError: If ``sample_dim`` is not a dimension of ``da``.
    """
    if sample_dim not in da.dims:
        label = f"variable {da.name!r}" if da.name is not None else "DataArray"
        raise ValueError(
            f"sample_dim={sample_dim!r} not found on {label} with dims={da.dims}."
        )
    feature_dims = tuple(d for d in da.dims if d != sample_dim)
    return ArrayLayout(
        sample_dim=sample_dim,
        dims=tuple(da.dims),
        feature_dims=feature_dims,
        feature_shape=tuple(da.sizes[d] for d in feature_dims),
        feature_coords=_coords_on(da, set(feature_dims)),
        name=da.name,
        attrs=dict(da.attrs),
    )


def dataset_layout(ds: xr.Dataset, sample_dim: Hashable) -> DatasetLayout:
    """Record the feature grid of every data variable in ``ds``.

    Args:
        ds: The Dataset to describe.
        sample_dim: The dimension indexing samples; every data variable
            must carry it.

    Returns:
        The Dataset's :class:`DatasetLayout`.

    Raises:
        ValueError: If ``ds`` has no data variables, or a data variable
            lacks ``sample_dim``.
    """
    if not ds.data_vars:
        raise ValueError("Cannot stack an empty Dataset (no data variables).")
    names = tuple(ds.data_vars)
    arrays = tuple(array_layout(ds[name], sample_dim) for name in names)
    bounds = [0]
    for arr in arrays:
        bounds.append(bounds[-1] + arr.n_features)
    return DatasetLayout(
        sample_dim=sample_dim,
        variables=names,
        arrays=arrays,
        bounds=tuple(bounds),
        attrs=dict(ds.attrs),
    )


def _match_index(
    da: xr.DataArray, dim: Hashable, expected: pd.Index, *, where: str
) -> xr.DataArray:
    """Reorder ``da`` along ``dim`` to ``expected``, or raise if impossible."""
    current = da.indexes[dim]
    if current.equals(expected):
        return da
    if (
        len(current) == len(expected)
        and current.is_unique
        and current.sort_values().equals(expected.sort_values())
    ):
        # Same labels in a different order (e.g. descending vs ascending
        # latitude) — reorder to the fit-time columns.
        return da.reindex({dim: expected})
    raise ValueError(
        f"{where}: coordinate {dim!r} does not match the one seen at fit time "
        f"(expected {_preview(expected)}, got {_preview(current)}). The "
        "estimator's feature columns are tied to the fit-time grid; select or "
        "regrid the input onto it first."
    )


def _preview(index: pd.Index) -> str:
    values = list(index[:3])
    tail = ", …" if len(index) > 3 else ""
    return f"[{', '.join(repr(v) for v in values)}{tail}] (n={len(index)})"


def conform_array(
    da: xr.DataArray, layout: ArrayLayout, *, where: str = "X"
) -> xr.DataArray:
    """Check ``da`` against a fit-time layout and put it in fit-time order.

    Feature dims may arrive in any order and indexed feature coords may
    be permuted; both are reordered to match ``layout``. A feature dim
    without an index coordinate (on either side) is matched by size only.

    Args:
        da: The new input.
        layout: The layout recorded at fit time.
        where: Label used in error messages (e.g. ``"X"`` or
            ``"variable 'ssh'"``).

    Returns:
        ``da`` transposed to ``(sample_dim, *feature_dims)`` with feature
        coordinates in fit-time order.

    Raises:
        ValueError: If the sample dim is missing, the feature dims differ,
            a size differs, or an indexed coordinate holds different labels.
    """
    sample_dim = layout.sample_dim
    if sample_dim not in da.dims:
        raise ValueError(
            f"{where}: sample_dim={sample_dim!r} not found in dims={da.dims}."
        )
    got = {d for d in da.dims if d != sample_dim}
    if got != set(layout.feature_dims):
        raise ValueError(
            f"{where}: feature dims {sorted(map(str, got))} do not match the "
            f"fit-time feature dims {sorted(map(str, layout.feature_dims))}."
        )
    fit_indexes = layout.feature_coords.xindexes
    for dim, size in zip(layout.feature_dims, layout.feature_shape, strict=True):
        if da.sizes[dim] != size:
            raise ValueError(
                f"{where}: dimension {dim!r} has size {da.sizes[dim]} but the "
                f"estimator was fit with size {size}."
            )
        if dim in fit_indexes and dim in da.indexes:
            expected = layout.feature_coords[dim].to_index()
            da = _match_index(da, dim, expected, where=where)
    da = da.transpose(sample_dim, *layout.feature_dims)
    _check_auxiliary_coords(da, layout, where=where)
    return da


def _check_auxiliary_coords(
    da: xr.DataArray, layout: ArrayLayout, *, where: str
) -> None:
    """Raise if a non-index feature coord on ``da`` differs from fit time.

    Curvilinear grids are identified by auxiliary coords such as 2-D
    ``nav_lat(y, x)``; two grids with the same ``y``/``x`` sizes and indexes
    would otherwise pass as identical. A coord the input does not carry is
    not checked — there is nothing to compare.
    """
    for name, fit_coord in layout.feature_coords.items():
        if name in layout.feature_coords.xindexes or name not in da.coords:
            continue
        got = np.asarray(da.coords[name].transpose(*fit_coord.dims).values)
        expected = np.asarray(fit_coord.values)
        if got.shape == expected.shape and _values_equal(got, expected):
            continue
        raise ValueError(
            f"{where}: auxiliary coordinate {name!r} does not match the one "
            "seen at fit time. The estimator's feature columns are tied to the "
            "fit-time grid; select or regrid the input onto it first."
        )


def _values_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, rtol=1e-12, atol=0.0, equal_nan=True))
    return bool(np.array_equal(a, b))


def conform_dataset(
    ds: xr.Dataset, layout: DatasetLayout, *, where: str = "X"
) -> xr.Dataset:
    """Check every variable of ``ds`` against a fit-time Dataset layout.

    Args:
        ds: The new input.
        layout: The layout recorded at fit time.
        where: Label prefix used in error messages.

    Returns:
        A Dataset holding exactly ``layout.variables``, each conformed via
        :func:`conform_array`.

    Raises:
        ValueError: If the data variables differ from the fit-time ones or
            any variable fails :func:`conform_array`.
    """
    got = set(ds.data_vars)
    if got != set(layout.variables):
        raise ValueError(
            f"{where}: data variables {sorted(map(str, got))} do not match the "
            f"fit-time variables {sorted(map(str, layout.variables))}."
        )
    conformed = {
        name: conform_array(ds[name], arr, where=f"{where} variable {name!r}")
        for name, arr in zip(layout.variables, layout.arrays, strict=True)
    }
    return xr.Dataset(conformed, attrs=ds.attrs)


def tree_layout(nodes: dict[str, xr.Dataset], sample_dim: Hashable) -> TreeLayout:
    """Record the feature grid of DataTree nodes joined column-wise.

    Args:
        nodes: ``{path: node Dataset}`` (see :func:`xrsklearn._src.tree.select_nodes`).
        sample_dim: The shared sample dimension.

    Returns:
        The nodes' :class:`TreeLayout`.
    """
    paths = tuple(nodes)
    datasets = tuple(dataset_layout(nodes[p], sample_dim) for p in paths)
    bounds = [0]
    for lay in datasets:
        bounds.append(bounds[-1] + lay.n_features)
    return TreeLayout(
        sample_dim=sample_dim, paths=paths, datasets=datasets, bounds=tuple(bounds)
    )


def conform_tree(
    nodes: dict[str, xr.Dataset], layout: TreeLayout
) -> dict[str, xr.Dataset]:
    """Check DataTree nodes against a fit-time :class:`TreeLayout`.

    The nodes must be exactly ``layout.paths`` and share one sample axis;
    a node whose sample labels are a permutation of the first node's is
    reordered to match.

    Args:
        nodes: ``{path: node Dataset}``.
        layout: The layout recorded at fit time.

    Returns:
        ``{path: conformed Dataset}`` in ``layout.paths`` order.

    Raises:
        ValueError: If the node paths differ, the sample axes cannot be
            matched, or any node fails :func:`conform_dataset`.
    """
    if set(nodes) != set(layout.paths):
        raise ValueError(
            f"DataTree nodes {sorted(nodes)} do not match the fit-time nodes "
            f"{sorted(layout.paths)}."
        )
    sample_dim = layout.sample_dim
    first = nodes[layout.paths[0]]
    reference = first.indexes.get(sample_dim)
    out: dict[str, xr.Dataset] = {}
    for path, lay in zip(layout.paths, layout.datasets, strict=True):
        ds = nodes[path]
        where = f"node {path!r}"
        if sample_dim not in ds.dims or ds.sizes[sample_dim] != first.sizes[sample_dim]:
            raise ValueError(
                f"{where}: all nodes must share the sample axis {sample_dim!r} "
                "for tree_mode='concat_features'."
            )
        if reference is not None and sample_dim in ds.indexes:
            current = ds.indexes[sample_dim]
            if not current.equals(reference):
                if current.is_unique and current.sort_values().equals(
                    reference.sort_values()
                ):
                    ds = ds.reindex({sample_dim: reference})
                else:
                    raise ValueError(
                        f"{where}: sample coordinate {sample_dim!r} differs from "
                        f"node {layout.paths[0]!r}; align the nodes first."
                    )
        out[path] = conform_dataset(ds, lay, where=where)
    return out


def to_2d(
    obj: xr.DataArray | xr.Dataset | dict[str, xr.Dataset], layout: Layout
) -> np.ndarray:
    """Flatten a conformed input to ``(n_samples, n_features)``.

    Args:
        obj: Output of :func:`conform_array` / :func:`conform_dataset`
            (or any input already in ``(sample_dim, *feature_dims)`` order).
        layout: The layout ``obj`` was conformed to.

    Returns:
        A 2-D numpy array. Lazy (dask) inputs are computed here.
    """
    if isinstance(layout, TreeLayout):
        assert isinstance(obj, dict)
        return np.concatenate(
            [
                to_2d(obj[p], lay)
                for p, lay in zip(layout.paths, layout.datasets, strict=True)
            ],
            axis=1,
        )
    if isinstance(layout, DatasetLayout):
        assert isinstance(obj, xr.Dataset)
        blocks = [
            to_2d(obj[name], arr)
            for name, arr in zip(layout.variables, layout.arrays, strict=True)
        ]
        return np.concatenate(blocks, axis=1)
    assert isinstance(obj, xr.DataArray)
    values = np.asarray(obj.values)
    return values.reshape(values.shape[0], layout.n_features)


def grid_from_2d(
    arr: np.ndarray,
    layout: ArrayLayout,
    samples: xr.Coordinates,
    *,
    dims: Sequence[Hashable] | None = None,
    name: Hashable | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Rebuild an N-D array on the fit-time feature grid (inverse of :func:`to_2d`).

    Args:
        arr: ``(n_samples, layout.n_features)`` array.
        layout: The feature grid to restore.
        samples: Sample-axis coordinates (see :func:`sample_coords`).
        dims: Output dim order; defaults to ``layout.dims``.
        name: Output name; defaults to ``layout.name``.
        attrs: Output attrs; defaults to ``layout.attrs``.

    Returns:
        The labeled array, with feature coords (and their attrs) from the
        layout and sample coords from ``samples``.
    """
    n_samples = arr.shape[0]
    out = xr.DataArray(
        arr.reshape(n_samples, *layout.feature_shape),
        dims=(layout.sample_dim, *layout.feature_dims),
        name=layout.name if name is None else name,
        attrs=dict(layout.attrs if attrs is None else attrs),
    )
    out = out.assign_coords(layout.feature_coords).assign_coords(samples)
    return out.transpose(*(layout.dims if dims is None else dims))


def generic_from_2d(
    arr: np.ndarray,
    sample_dim: Hashable,
    samples: xr.Coordinates,
    *,
    new_feature_dim: Hashable,
    feature_coord: np.ndarray | None = None,
    name: Hashable | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Label an output that no longer lives on the input feature grid.

    ``(n_samples,)`` becomes ``(sample_dim,)``; ``(n_samples, k)`` becomes
    ``(sample_dim, new_feature_dim)`` with an integer ``new_feature_dim``
    coordinate.

    Args:
        arr: 1-D or 2-D estimator output.
        sample_dim: The sample dimension.
        samples: Sample-axis coordinates.
        new_feature_dim: Name of the output feature dimension.
        feature_coord: Labels for ``new_feature_dim``; defaults to
            ``0 … k-1``.
        name: Output name.
        attrs: Output attrs.

    Returns:
        The labeled array.

    Raises:
        ValueError: If ``arr`` is neither 1-D nor 2-D.
    """
    if arr.ndim == 1:
        out = xr.DataArray(arr, dims=(sample_dim,), name=name, attrs=attrs or {})
    elif arr.ndim == 2:
        out = xr.DataArray(
            arr,
            dims=(sample_dim, new_feature_dim),
            coords={
                new_feature_dim: np.arange(arr.shape[1])
                if feature_coord is None
                else feature_coord
            },
            name=name,
            attrs=attrs or {},
        )
    else:
        raise ValueError(
            f"Cannot label sklearn output with shape {arr.shape}; expected 1-D or 2-D."
        )
    # A sample coord named like (or living on) the new output dim would clash
    # with the generated component labels.
    clash = [
        name
        for name, coord in samples.items()
        if name == new_feature_dim or new_feature_dim in coord.dims
    ]
    return out.assign_coords(samples.to_dataset().drop_vars(clash).coords)


def dataset_from_2d(
    arr: np.ndarray,
    layout: DatasetLayout,
    samples: xr.Coordinates,
    *,
    dims: dict[Hashable, tuple[Hashable, ...]] | None = None,
) -> xr.Dataset:
    """Rebuild a Dataset on the fit-time grid (inverse of :func:`to_2d`).

    Args:
        arr: ``(n_samples, layout.n_features)`` array.
        layout: The Dataset layout to restore; ``layout.bounds`` says which
            columns belong to which variable.
        samples: Sample-axis coordinates.
        dims: Per-variable output dim order; defaults to each variable's
            fit-time order.

    Returns:
        A Dataset with one variable per ``layout.variables`` entry, each on
        its own fit-time grid, and the fit-time Dataset attrs.
    """
    dims = dims or {}
    variables = {
        name: grid_from_2d(arr[:, lo:hi], sub, samples, dims=dims.get(name))
        for name, sub, lo, hi in zip(
            layout.variables,
            layout.arrays,
            layout.bounds[:-1],
            layout.bounds[1:],
            strict=True,
        )
    }
    return xr.Dataset(variables, attrs=dict(layout.attrs))


def tree_from_2d(
    arr: np.ndarray,
    layout: TreeLayout,
    samples: xr.Coordinates,
    *,
    dims: dict[str, dict[Hashable, tuple[Hashable, ...]]] | None = None,
) -> xr.DataTree:
    """Rebuild DataTree nodes on the fit-time grid (inverse of :func:`to_2d`).

    Args:
        arr: ``(n_samples, layout.n_features)`` array.
        layout: The tree layout to restore.
        samples: Sample-axis coordinates.
        dims: Per-node, per-variable output dim order.

    Returns:
        A DataTree with one node per ``layout.paths`` entry.
    """
    dims = dims or {}
    nodes = {
        path: dataset_from_2d(arr[:, lo:hi], lay, samples, dims=dims.get(path))
        for path, lay, lo, hi in zip(
            layout.paths,
            layout.datasets,
            layout.bounds[:-1],
            layout.bounds[1:],
            strict=True,
        )
    }
    return xr.DataTree.from_dict(nodes)


def matches(obj: xr.DataArray | xr.Dataset | xr.DataTree, layout: Layout) -> bool:
    """Whether ``obj`` plausibly lives on ``layout``'s feature grid.

    A cheap structural check (same kind, same feature dims / variables) used
    to decide whether an input is in the fit-time feature space at all —
    e.g. ``inverse_transform`` receives either grid-shaped data (from a
    scaler) or component scores (from PCA). A match is then verified in
    full by :func:`conform_array` / :func:`conform_dataset`.

    Args:
        obj: The input.
        layout: A fit-time layout.

    Returns:
        ``True`` if ``obj`` has the layout's kind and feature structure.
    """
    if isinstance(layout, TreeLayout):
        return isinstance(obj, xr.DataTree) and {
            node.path for node in obj.subtree if node.data_vars
        } >= set(layout.paths)
    if isinstance(layout, DatasetLayout):
        return (
            isinstance(obj, xr.Dataset)
            and set(obj.data_vars) == set(layout.variables)
            and all(
                _array_matches(obj[name], arr)
                for name, arr in zip(layout.variables, layout.arrays, strict=True)
            )
        )
    return isinstance(obj, xr.DataArray) and _array_matches(obj, layout)


def _array_matches(da: xr.DataArray, layout: ArrayLayout) -> bool:
    """Dim names *and* sizes match, and so do the labels of shared indexes.

    Names alone are not enough: PCA scores on ``(time, component)`` share
    their dim names with training data that had a ``component`` feature dim.
    """
    if layout.sample_dim not in da.dims:
        return False
    if {d for d in da.dims if d != layout.sample_dim} != set(layout.feature_dims):
        return False
    fit_indexes = layout.feature_coords.xindexes
    for dim, size in zip(layout.feature_dims, layout.feature_shape, strict=True):
        if da.sizes[dim] != size:
            return False
        if dim in fit_indexes and dim in da.indexes:
            got = da.indexes[dim]
            expected = layout.feature_coords[dim].to_index()
            if not got.sort_values().equals(expected.sort_values()):
                return False
    return True
