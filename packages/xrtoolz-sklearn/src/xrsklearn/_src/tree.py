"""DataTree support for :class:`~xrsklearn.XarrayEstimator`.

A DataTree holds several Datasets at once — resolutions, regions,
ensemble members, train/validation groups — so "fit an estimator on a
tree" has more than one meaning. ``tree_mode`` selects it:

``"per_node"`` (default)
    One independently fitted estimator per data node, stored in
    ``estimators_`` by path. Nodes may have different grids (coarse and
    fine resolutions, different regions). Every verb maps each node to
    its own estimator and returns a tree with the same structure.

``"pool_samples"``
    One estimator fitted on the samples of every node stacked along
    ``sample_dim``. All nodes must share the same feature grid (ensemble
    members, years, train groups). Verbs apply that one estimator to each
    node separately and return a tree; ``score`` pools the nodes.

``"concat_features"``
    One estimator whose features are the variables of every node side by
    side, like the variables of a single Dataset. All nodes must share the
    sample axis (predictors at several resolutions for one target). Outputs
    on the input grid (one-to-one transforms, ``inverse_transform``) come
    back as a tree; reduced outputs (PCA scores) as one DataArray. This
    mode is implemented by :class:`~xrsklearn._src.layout.TreeLayout`
    inside the ordinary marshalling path.

Which nodes take part: every node with data variables, or exactly
``tree_paths`` when given. Each node is read with its inherited
coordinates, so a ``time`` coordinate stored once on the root reaches
every node.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from sklearn.base import clone

from xrsklearn._src.layout import (
    array_layout,
    conform_array,
    conform_dataset,
    dataset_layout,
)


if TYPE_CHECKING:
    from xrsklearn._src.wrap import XarrayEstimator


#: What fitting on a DataTree means (see the module docstring).
TreeMode = Literal["per_node", "pool_samples", "concat_features"]

#: Variable name given to unnamed DataArray results when they become tree nodes.
RESULT_NAMES: dict[str, str] = {
    "transform": "transformed",
    "fit_transform": "transformed",
    "inverse_transform": "reconstructed",
    "predict": "prediction",
    "predict_proba": "probability",
}


def _norm(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def select_nodes(
    tree: xr.DataTree, paths: Sequence[str] | None = None
) -> dict[str, xr.Dataset]:
    """The nodes of ``tree`` that take part, as ``{path: Dataset}``.

    Args:
        tree: The input tree.
        paths: Node paths to use (with or without a leading ``/``); by
            default every node that has data variables.

    Returns:
        Each selected node as a Dataset including inherited coordinates,
        keyed by absolute path, in tree order (or ``paths`` order).

    Raises:
        KeyError: If a requested path does not exist.
        ValueError: If no node (or a requested node) has data variables.
    """
    if paths is None:
        nodes = {
            n.path: n.to_dataset(inherit=True) for n in tree.subtree if n.data_vars
        }
        if not nodes:
            raise ValueError("DataTree has no node with data variables.")
        return nodes
    out: dict[str, xr.Dataset] = {}
    for path in paths:
        path = _norm(path)
        try:
            node = tree[path]
        except KeyError as exc:
            raise KeyError(f"DataTree has no node {path!r}.") from exc
        assert isinstance(node, xr.DataTree)
        if not node.data_vars:
            raise ValueError(f"DataTree node {path!r} has no data variables.")
        out[path] = node.to_dataset(inherit=True)
    return out


def node_targets(
    y: Any, paths: Sequence[str]
) -> dict[str, xr.DataArray | xr.Dataset | None]:
    """Split a per-node target into ``{path: target}``.

    Args:
        y: ``None``, a DataTree with a node per path, or a mapping from
            path to target. A node with a single data variable yields that
            DataArray; otherwise its Dataset.
        paths: The paths that need a target.

    Returns:
        ``{path: target or None}``.

    Raises:
        TypeError: If ``y`` is not per-node (e.g. one NumPy array for
            several independent estimators).
        KeyError: If a path has no target.
    """
    if y is None:
        return dict.fromkeys(paths)
    if isinstance(y, xr.DataTree):
        mapping: dict[str, Any] = {}
        for p in paths:
            try:
                node = y[p]
            except KeyError:
                continue
            if isinstance(node, xr.DataTree) and node.data_vars:
                mapping[p] = _as_target(node.to_dataset(inherit=True))
    elif isinstance(y, Mapping) and not isinstance(y, xr.Dataset):
        mapping = {_norm(str(k)): v for k, v in y.items()}
    else:
        raise TypeError(
            "With a DataTree X, y must be None, a DataTree, or a mapping from "
            f"node path to target; got {type(y).__name__}."
        )
    missing = [p for p in paths if p not in mapping]
    if missing:
        raise KeyError(f"y has no target for node(s) {missing}.")
    return {p: mapping[p] for p in paths}


def _as_target(ds: xr.Dataset) -> xr.DataArray | xr.Dataset:
    names = list(ds.data_vars)
    return ds[names[0]] if len(names) == 1 else ds


def to_tree(
    results: Mapping[str, Any], method: str, base: xr.DataTree | None = None
) -> xr.DataTree:
    """Assemble per-node results into a DataTree.

    Args:
        results: ``{path: DataArray | Dataset}``.
        method: The verb that produced them (names unnamed DataArrays).
        base: The input tree. When given, the result keeps its structure:
            nodes that took no part (excluded by ``tree_paths``, metadata
            or coordinate-only nodes) are carried over unchanged.

    Returns:
        A DataTree with one node per path (plus ``base``'s other nodes).

    Raises:
        TypeError: If a result is not a DataArray or Dataset.
    """
    nodes: dict[str, xr.Dataset] = {}
    for path, res in results.items():
        if isinstance(res, xr.DataArray):
            name = (
                res.name if res.name is not None else RESULT_NAMES.get(method, method)
            )
            res = res.to_dataset(name=name)
        if not isinstance(res, xr.Dataset):
            raise TypeError(
                f"{method} returned {type(res).__name__} for node {path!r}; only "
                "DataArray / Dataset results can be assembled into a DataTree."
            )
        nodes[path] = res
    if base is not None:
        kept = {n.path: n.to_dataset(inherit=False) for n in base.subtree}
        nodes = {**kept, **nodes}
    return xr.DataTree.from_dict(nodes)


# ---------- per_node -----------------------------------------------------


def _node_estimator(est: XarrayEstimator) -> XarrayEstimator:
    """An unfitted single-node wrapper with ``est``'s settings."""
    from xrsklearn._src.wrap import XarrayEstimator

    return XarrayEstimator(
        clone(est.estimator),
        sample_dim=est.sample_dim,
        new_feature_dim=est.new_feature_dim,
        nan_policy=est.nan_policy,
        missing=est.missing,
    )


def fit_per_node(
    est: XarrayEstimator,
    tree: xr.DataTree,
    y: Any,
    kwargs: dict[str, Any],
    *,
    transform: bool,
) -> xr.DataTree | None:
    """Fit one wrapper per node; return the transformed tree if asked."""
    nodes = select_nodes(tree, est.tree_paths)
    targets = node_targets(y, list(nodes))
    fitted: dict[str, XarrayEstimator] = {}
    results: dict[str, Any] = {}
    for path, ds in nodes.items():
        sub = _node_estimator(est)
        if transform:
            results[path] = sub.fit_transform(ds, targets[path], **kwargs)
        else:
            sub.fit(ds, targets[path], **kwargs)
        fitted[path] = sub
    est.estimators_ = fitted
    est.tree_paths_ = tuple(nodes)
    return to_tree(results, "fit_transform", tree) if transform else None


def apply_per_node(est: XarrayEstimator, method: str, tree: xr.DataTree) -> xr.DataTree:
    """Run ``method`` on every fitted node with that node's own estimator."""
    nodes = select_nodes(tree, est.tree_paths_)
    return to_tree(
        {p: getattr(est.estimators_[p], method)(ds) for p, ds in nodes.items()},
        method,
        tree,
    )


def score_per_node(est: XarrayEstimator, tree: xr.DataTree, y: Any) -> dict[str, float]:
    """``{path: score}`` from each node's own estimator."""
    nodes = select_nodes(tree, est.tree_paths_)
    targets = node_targets(y, list(nodes))
    return {p: est.estimators_[p].score(ds, targets[p]) for p, ds in nodes.items()}


# ---------- pool_samples -------------------------------------------------


def pool(
    nodes: Mapping[str, xr.Dataset], sample_dim: Hashable
) -> tuple[xr.Dataset, list[int]]:
    """Stack node Datasets along ``sample_dim``.

    Every node is conformed to the first node's feature grid, so nodes on
    different grids fail with an error naming the node rather than being
    outer-joined into NaN.

    Returns:
        ``(pooled Dataset, per-node sample counts)``.
    """
    paths = list(nodes)
    reference = dataset_layout(nodes[paths[0]], sample_dim)
    parts = [conform_dataset(nodes[p], reference, where=f"node {p!r}") for p in paths]
    pooled = xr.concat(
        parts, dim=sample_dim, coords="minimal", compat="override", join="override"
    )
    return pooled, [part.sizes[sample_dim] for part in parts]


def _pool_targets(
    y: Any, paths: Sequence[str], sample_dim: Hashable
) -> xr.DataArray | xr.Dataset | np.ndarray | None:
    if y is None or isinstance(y, np.ndarray):
        return y  # already pooled (or absent)
    targets = [t for t in node_targets(y, paths).values() if t is not None]
    if all(isinstance(t, xr.DataArray | xr.Dataset) for t in targets):
        # Conform every target to the first one's layout so permuted target
        # coords are reordered and mismatched ones raise, rather than being
        # overridden positionally.
        first = targets[0]
        if isinstance(first, xr.Dataset):
            ref = dataset_layout(first, sample_dim)
            parts = [
                conform_dataset(t, ref, where=f"y node {p!r}")
                for p, t in zip(paths, targets, strict=True)
            ]
        else:
            ref_arr = array_layout(first, sample_dim)
            parts = [
                conform_array(t, ref_arr, where=f"y node {p!r}")
                for p, t in zip(paths, targets, strict=True)
            ]
        return xr.concat(
            parts, dim=sample_dim, coords="minimal", compat="override", join="override"
        )
    if any(isinstance(t, xr.DataArray | xr.Dataset) for t in targets):
        raise TypeError(
            "Pooled targets must be all xarray or all NumPy; got a mix across nodes."
        )
    return np.concatenate([np.asarray(t) for t in targets], axis=0)


def _split(result: Any, sample_dim: Hashable, sizes: Sequence[int]) -> list[Any]:
    bounds = np.cumsum([0, *sizes])
    return [
        result.isel({sample_dim: slice(int(lo), int(hi))})
        for lo, hi in pairwise(bounds)
    ]


def _restore_node_meta(result: Any, node: xr.Dataset) -> Any:
    """Give a split pooled result back its own node's scalar coords and attrs.

    Pooling keeps only the first node's scalar coords (an ensemble member
    id, say) and attrs; each split part must carry its own node's instead.
    """
    scalars = {k: v for k, v in node.coords.items() if v.ndim == 0}
    result = result.assign_coords(scalars)
    if isinstance(result, xr.Dataset):
        result.attrs = dict(node.attrs)
        for name in result.data_vars:
            if name in node.data_vars:
                result[name].attrs = dict(node[name].attrs)
    return result


def fit_pooled(
    est: XarrayEstimator,
    tree: xr.DataTree,
    y: Any,
    kwargs: dict[str, Any],
    *,
    transform: bool,
) -> xr.DataTree | None:
    """Fit ``est`` on all nodes' samples; return the transformed tree if asked."""
    from xrsklearn._src.wrap import XarrayEstimator

    nodes = select_nodes(tree, est.tree_paths)
    sample_dim = est._resolve_sample_dim(next(iter(nodes.values())))
    pooled, sizes = pool(nodes, sample_dim)
    y_pooled = _pool_targets(y, list(nodes), sample_dim)
    if transform:
        out = XarrayEstimator.fit_transform(est, pooled, y_pooled, **kwargs)
        results = {
            path: _restore_node_meta(part, nodes[path])
            for path, part in zip(nodes, _split(out, sample_dim, sizes), strict=True)
        }
    else:
        XarrayEstimator.fit(est, pooled, y_pooled, **kwargs)
    est.tree_paths_ = tuple(nodes)
    return to_tree(results, "fit_transform", tree) if transform else None


def apply_pooled(est: XarrayEstimator, method: str, tree: xr.DataTree) -> xr.DataTree:
    """Run ``method`` of the one pooled estimator on every node separately."""
    nodes = select_nodes(tree, est.tree_paths)
    return to_tree(
        {p: getattr(est, method)(ds) for p, ds in nodes.items()}, method, tree
    )


def score_pooled(est: XarrayEstimator, tree: xr.DataTree, y: Any) -> float:
    """One score over the pooled samples of every node."""
    nodes = select_nodes(tree, est.tree_paths)
    sample_dim = est._resolve_sample_dim(next(iter(nodes.values())))
    pooled, _ = pool(nodes, sample_dim)
    return est.score(pooled, _pool_targets(y, list(nodes), sample_dim))
