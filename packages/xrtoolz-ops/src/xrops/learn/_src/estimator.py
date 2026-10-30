"""`XarrayEstimator`: a scikit-learn estimator on DataArrays, Datasets and DataTrees.

The fitting, the NaN policies and the fitted state are pipekit-array's
`pipekit_array.Estimator` (geostack OP-O3); this module only adds what is
xarray's own: a Dataset's variables side by side as one feature table, and
a DataTree fitted per node or pooled over its nodes.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, ClassVar, Literal, Self, get_args

import xarray as xr
from pipekit_array import DimArray, Estimator, to_rows
from pipekit_array.estimator import NanPolicy

from xrops._src.operator import Operator
from xrops.learn._src.layout import (
    SAMPLE,
    Layout,
    Samples,
    Variable,
    concat_tables,
    pack,
    pack_target,
    unpack,
)


#: What fitting on a DataTree means.
TreeMode = Literal["per_node", "pool_samples"]

#: Variable names given to unnamed results that become DataTree nodes.
_RESULT_NAMES: dict[str, str] = {
    "transform": "transformed",
    "inverse_transform": "reconstructed",
    "predict": "prediction",
    "predict_proba": "probability",
}


def _nodes(tree: xr.DataTree) -> dict[str, xr.Dataset]:
    """Every node with data variables, with its inherited coordinates."""
    nodes = {n.path: n.to_dataset(inherit=True) for n in tree.subtree if n.data_vars}
    if not nodes:
        raise ValueError(
            "XarrayEstimator: the DataTree has no node with data variables."
        )
    return nodes


def _node_target(y: Any, path: str) -> xr.DataArray | None:
    if y is None:
        return None
    if isinstance(y, xr.DataTree):
        node = y[path]
        assert isinstance(node, xr.DataTree)
        names = list(node.data_vars)
        if len(names) != 1:
            raise ValueError(
                f"XarrayEstimator: the target node {path!r} needs one data variable; "
                f"got {names}."
            )
        return node.to_dataset(inherit=True)[names[0]]
    if isinstance(y, Mapping):
        return y[path]
    raise TypeError(
        "XarrayEstimator: y for a DataTree is a DataTree or {path: DataArray}."
    )


def _on_fitted_grid(x: Any, sample_dims: tuple[Hashable, ...], layout: Layout) -> bool:
    """Whether ``x`` has the fitted variables and feature grids (a one-to-one output).

    Names and sizes both count: scores along a ``feature_dim`` that reuses a
    fitted feature dim's name still have another size.
    """

    def same(da: xr.DataArray, v: Variable) -> bool:
        features = {d: n for d, n in da.sizes.items() if d not in sample_dims}
        return features == dict(zip(v.feature_dims, v.feature_sizes, strict=True))

    if isinstance(x, xr.Dataset) and not layout.is_array:
        variables = {v.name: v for v in layout.variables}
        return set(x.data_vars) == set(variables) and all(
            same(x[name], v) for name, v in variables.items()
        )
    if isinstance(x, xr.DataArray) and layout.is_array:
        return same(x, layout.variables[0])
    return False


def _as_node(result: xr.DataArray | xr.Dataset, method: str) -> xr.Dataset:
    if isinstance(result, xr.Dataset):
        return result
    return result.to_dataset(
        name=result.name if result.name is not None else _RESULT_NAMES[method]
    )


class XarrayEstimator(Operator):
    """A scikit-learn-style estimator fitted and applied on xarray objects.

    The sample dims are chosen by name; every other dim of a variable is a
    feature. A Dataset's variables are put side by side as the columns of
    one table, so each may have its own feature grid. Outputs come back as:

    - one value per sample (``predict``): a DataArray over the sample dims;
    - the fitted features (a scaler, ``inverse_transform``): the input's
      variables on their own grids, as a DataArray or a Dataset;
    - a new output dim (PCA scores, ``predict_proba``): one DataArray with
      ``feature_dim`` where the first variable's feature dims were.

    Args:
        estimator: Any object with scikit-learn's ``fit`` and the methods
            you call. It is cloned on ``fit``; the original is never fitted.
        sample_dims: The dim (or dims) whose entries are samples.
        feature_dim: The name of a new output dim, for estimators whose
            output columns are not the input's features (``"component"``
            for PCA). ``predict_proba`` defaults to ``"class"``.
        nan_policy: ``"propagate"``, ``"raise"``, ``"mask_samples"`` or
            ``"mask_features"``, as in `pipekit_array.Estimator`.
        method: What calling the operator runs once fitted:
            ``"transform"``, ``"predict"`` or ``"predict_proba"``.
        tree_mode: What fitting on a DataTree means: ``"per_node"`` fits
            one estimator per node (nodes may have different grids);
            ``"pool_samples"`` fits one on every node's samples (nodes share
            their variables and feature grids).

    Attributes:
        estimator_: The fitted scikit-learn estimator (``None`` after a
            ``"per_node"`` DataTree fit; see ``estimators_``).
        estimators_: ``{path: fitted XarrayEstimator}`` after a
            ``"per_node"`` DataTree fit.

    Raises:
        ValueError: An unknown ``tree_mode``, ``nan_policy`` or ``method``.

    Example:
        Decompose a ``(time, lat, lon)`` cube with PCA and recover the grid:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.decomposition import PCA
        >>> rng = np.random.default_rng(0)
        >>> da = xr.DataArray(rng.normal(size=(8, 3, 4)), dims=("time", "lat", "lon"))
        >>> est = XarrayEstimator(PCA(3), sample_dims="time", feature_dim="component")
        >>> scores = est.fit(da).transform(da)
        >>> scores.dims, scores.shape
        (('time', 'component'), (8, 3))
        >>> est.inverse_transform(scores).dims
        ('time', 'lat', 'lon')
        >>> est.estimator_.components_.shape
        (3, 12)

        ```

        Scale every variable of a Dataset on its own grid:

        ```pycon
        >>> from sklearn.preprocessing import StandardScaler
        >>> ds = xr.Dataset({"u": da, "v": da.isel(lat=0)})
        >>> scaler = XarrayEstimator(StandardScaler(), sample_dims="time")
        >>> out = scaler.fit(ds).transform(ds)
        >>> {name: out[name].dims for name in out.data_vars}
        {'u': ('time', 'lat', 'lon'), 'v': ('time', 'lon')}

        ```
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_exclude__: ClassVar[tuple[str, ...]] = (
        "fitted_",
        "layout_",
        "estimators_",
    )

    def __init__(
        self,
        estimator: Any,
        *,
        sample_dims: Hashable | Sequence[Hashable],
        feature_dim: str | None = None,
        nan_policy: NanPolicy = "propagate",
        method: str = "transform",
        tree_mode: TreeMode = "per_node",
    ) -> None:
        if tree_mode not in get_args(TreeMode):
            raise ValueError(
                f"tree_mode must be one of {get_args(TreeMode)}; got {tree_mode!r}"
            )
        self.estimator = estimator
        self.sample_dims = (
            (sample_dims,)
            if isinstance(sample_dims, str) or not isinstance(sample_dims, Sequence)
            else tuple(sample_dims)
        )
        self.feature_dim = feature_dim
        self.nan_policy = nan_policy
        self.method = method
        self.tree_mode = tree_mode
        self._table_estimator()  # validates nan_policy and method now
        self.fitted_: Estimator | None = None
        self.layout_: Layout | None = None
        self.estimators_: dict[str, XarrayEstimator] | None = None

    def _table_estimator(self) -> Estimator:
        return Estimator(
            self.estimator,
            sample_dims=(SAMPLE,),
            feature_dim=self.feature_dim,
            nan_policy=self.nan_policy,
            method=self.method,
        )

    def _node_estimator(self) -> XarrayEstimator:
        return XarrayEstimator(
            self.estimator,
            sample_dims=self.sample_dims,
            feature_dim=self.feature_dim,
            nan_policy=self.nan_policy,
            method=self.method,
        )

    @property
    def estimator_(self) -> Any:
        """The fitted scikit-learn estimator."""
        return None if self.fitted_ is None else self.fitted_.estimator_

    # ------------------------------------------------------------------ fit

    def fit(self, x: xr.DataArray | xr.Dataset | xr.DataTree, y: Any = None) -> Self:
        """Learn the estimator from ``x`` (and ``y``, for a supervised one).

        Args:
            x: A DataArray, Dataset or DataTree with the sample dims.
            y: Targets: a DataArray over the sample dims (plus at most one
                target dim); for a DataTree, a DataTree with one variable
                per node or ``{path: DataArray}``.

        Returns:
            This operator, fitted.

        Raises:
            ValueError: A variable lacks a sample dim, or (``"pool_samples"``)
                the nodes do not share their feature grids.
        """
        self.fitted_ = self.layout_ = self.estimators_ = None
        if isinstance(x, xr.DataTree):
            nodes = _nodes(x)
            if self.tree_mode == "per_node":
                self.estimators_ = {
                    path: self._node_estimator().fit(ds, _node_target(y, path))
                    for path, ds in nodes.items()
                }
                return self
            tables, targets, layout = [], [], None
            for path, ds in nodes.items():
                table, layout, samples = pack(ds, self.sample_dims, layout)
                tables.append(table)
                target = _node_target(y, path)
                if target is not None:
                    targets.append(pack_target(target, samples))
            assert layout is not None
            self._fit_table(
                concat_tables(tables),
                concat_tables(targets) if targets else None,
                layout,
            )
            return self
        table, layout, samples = pack(x, self.sample_dims)
        self._fit_table(table, None if y is None else pack_target(y, samples), layout)
        return self

    def _fit_table(
        self, table: DimArray, target: DimArray | None, layout: Layout
    ) -> None:
        self.fitted_ = self._table_estimator().fit(table, target)
        self.layout_ = layout

    # ------------------------------------------------------------- apply

    def transform(self, x: xr.DataArray | xr.Dataset | xr.DataTree) -> Any:
        """The estimator's ``transform`` of ``x``, rebuilt as xarray."""
        return self._run("transform", x)

    def predict(self, x: xr.DataArray | xr.Dataset | xr.DataTree) -> Any:
        """One prediction per sample: a DataArray over the sample dims."""
        return self._run("predict", x)

    def predict_proba(self, x: xr.DataArray | xr.Dataset | xr.DataTree) -> Any:
        """Class probabilities per sample, along ``feature_dim`` (``"class"``)."""
        return self._run("predict_proba", x)

    def inverse_transform(self, z: xr.DataArray | xr.Dataset | xr.DataTree) -> Any:
        """Map ``transform``'s output back to the fitted variables and grids.

        Args:
            z: ``transform``'s output: a DataArray with the sample dims and
                one output dim (scores); or, for a one-to-one transform (a
                scaler), the fitted variables on their grids, as a DataArray
                or Dataset; or a DataTree of either.

        Returns:
            The fitted variables on their grids; features dropped by
            ``"mask_features"`` are NaN.
        """
        return self._run("inverse_transform", z)

    def fit_transform(
        self, x: xr.DataArray | xr.Dataset | xr.DataTree, y: Any = None
    ) -> Any:
        """``fit(x, y)`` then ``transform(x)``.

        This is ``transform`` of the fitted estimator, not scikit-learn's
        ``fit_transform``; the two differ for estimators that refit inside
        ``transform`` (NMF solves for W again, so its scores move by about
        1e-3).
        """
        return self.fit(x, y).transform(x)

    def _apply(self, x: Any) -> Any:
        return getattr(self, self.method)(x)

    def _apply_tree(self, x: Any) -> Any:
        # the whole tree at once: fitted per node or pooled, not leaf by leaf
        return self._apply(x)

    def _run(self, method: str, x: Any) -> Any:
        if isinstance(x, xr.DataTree):
            return self._run_tree(method, x)
        if self.estimators_ is not None:
            raise TypeError(
                "XarrayEstimator was fitted per node on a DataTree; pass one."
            )
        if self.fitted_ is None or self.layout_ is None:
            raise ValueError("XarrayEstimator is not fitted: call fit(x) first.")
        if method == "inverse_transform" and not _on_fitted_grid(
            x, self.sample_dims, self.layout_
        ):
            table, samples = self._pack_output(x)
        else:
            table, _, samples = pack(x, self.sample_dims, self.layout_)
        out = unpack(getattr(self.fitted_, method)(table), self.layout_, samples)
        classes = getattr(self.estimator_, "classes_", None)
        dim = self.feature_dim or "class"
        if method == "predict_proba" and classes is not None and dim in out.dims:
            # label the probability columns with the classes they belong to
            out = out.assign_coords({dim: list(classes)})
        return out

    def _pack_output(self, z: Any) -> tuple[DimArray, Samples]:
        """``transform``'s output (sample dims plus one dim) as a table."""
        if not isinstance(z, xr.DataArray):
            raise TypeError(
                f"inverse_transform takes a DataArray; got {type(z).__name__}."
            )
        order = tuple(d for d in z.dims if d in self.sample_dims)
        rest = [d for d in z.dims if d not in self.sample_dims]
        if len(order) != len(self.sample_dims) or len(rest) != 1:
            raise ValueError(
                f"inverse_transform takes the sample dims {self.sample_dims} plus one "
                f"output dim; got {tuple(z.dims)}."
            )
        rows, _ = to_rows(z.transpose(*order, *rest), order)
        samples = Samples(
            dims=order,
            sizes=tuple(z.sizes[d] for d in order),
            coords={k: c for k, c in z.coords.items() if set(c.dims) <= set(order)},
        )
        return DimArray(rows, (SAMPLE, str(rest[0]))), samples

    def _run_tree(self, method: str, tree: xr.DataTree) -> xr.DataTree:
        nodes = _nodes(tree)
        if self.estimators_ is not None:
            missing = sorted(set(self.estimators_) - set(nodes))
            if missing:
                raise ValueError(
                    f"XarrayEstimator: the DataTree lacks the fitted nodes {missing}."
                )
            estimators = self.estimators_
        else:
            estimators = dict.fromkeys(nodes, self)
        # nodes that take no part (metadata only, coordinates only, unfitted)
        # come through unchanged, with their own attrs
        out = {n.path: n.to_dataset(inherit=False) for n in tree.subtree}
        for path, est in estimators.items():
            ds = nodes[path]
            layout = est.layout_
            if method == "inverse_transform" and not (
                layout is not None and _on_fitted_grid(ds, est.sample_dims, layout)
            ):
                names = list(ds.data_vars)
                if len(names) != 1:
                    raise ValueError(
                        f"inverse_transform: node {path!r} needs one data variable; "
                        f"got {names}."
                    )
                arg: Any = ds[names[0]]
            else:
                arg = ds
            result = _as_node(est._run(method, arg), method)
            out[path] = result.assign_attrs({**out[path].attrs, **result.attrs})
        return xr.DataTree.from_dict(out)


__all__ = ["TreeMode", "XarrayEstimator"]
