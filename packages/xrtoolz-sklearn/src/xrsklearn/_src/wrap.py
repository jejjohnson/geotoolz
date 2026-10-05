"""Generic xarray ↔ scikit-learn bridge.

The :class:`XarrayEstimator` wraps any sklearn ``BaseEstimator`` and lets
it operate on N-D :class:`xr.DataArray` / :class:`xr.Dataset` inputs.
The data flow is **stack → delegate → unstack**:

    Phase 1 (stack):    xr.DataArray  →  (n_samples, n_features) numpy
    Phase 2 (delegate): sklearn.fit / .transform / .predict on numpy
    Phase 3 (unstack):  numpy result  →  xr.DataArray (re-gridded)

The wrapper does not monkey-patch sklearn and does not pass xarray
objects into the estimator — sklearn sees a plain 2-D numpy array. The
feature grid seen at fit time is recorded as a layout
(:mod:`xrsklearn._src.layout`); every later input is checked against it
and reordered to it, so the columns sklearn sees always mean what they
meant during ``fit``.

This is the foundation used by :mod:`xrtoolz.transforms.decompose` to
expose PCA / EOF / ICA / NMF / KMeans as thin presets.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard, get_args

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.utils.metaestimators import available_if

from xrsklearn._src.layout import (
    ArrayLayout,
    DatasetLayout,
    Layout,
    TreeLayout,
    array_layout,
    conform_array,
    conform_dataset,
    conform_tree,
    dataset_from_2d,
    dataset_layout,
    generic_from_2d,
    grid_from_2d,
    matches,
    sample_coords,
    to_2d,
    tree_from_2d,
    tree_layout,
)
from xrsklearn._src.nan import (
    MissingKind,
    NanPolicy,
    _rows,
    check_no_missing,
    check_policy,
    learn_feature_mask,
    learn_sample_mask,
    mask_fit_params,
    masks_features,
    masks_samples,
    missing_mask,
    restore_columns,
    restore_rows,
)
from xrsklearn._src.tree import (
    TreeMode,
    apply_per_node,
    apply_pooled,
    fit_per_node,
    fit_pooled,
    score_per_node,
    score_pooled,
    select_nodes,
)


#: Inputs ``X`` every verb accepts.
XInput = xr.DataArray | xr.Dataset | xr.DataTree | np.ndarray
#: Targets ``y`` (a DataTree or ``{path: target}`` mapping for tree inputs).
YInput = xr.DataArray | xr.Dataset | xr.DataTree | np.ndarray | Mapping[str, Any] | None
#: What the transform / predict verbs return.
Output = xr.DataArray | xr.Dataset | xr.DataTree | np.ndarray

#: Dimension name of ``predict_proba`` outputs, labeled by ``classes_``.
CLASS_DIM = "class"

# Methods exposed only when the wrapped estimator implements them.
_GATED_VERBS = frozenset(
    {
        "transform",
        "fit_transform",
        "inverse_transform",
        "predict",
        "predict_proba",
        "score",
    }
)


@dataclass
class _Batch:
    """One marshalled input: the 2-D array plus what is needed to label outputs."""

    arr: np.ndarray
    layout: Layout | None = None
    sample_dim: Hashable | None = None
    samples: xr.Coordinates | None = None
    sample_index: pd.Index | None = None
    in_dims: Any = None  # input dim order(s), shaped like the input
    name: Hashable | None = None
    attrs: dict[str, Any] | None = None
    valid: np.ndarray | None = None
    features: np.ndarray | None = None
    full: np.ndarray | None = None  # fit-time X before the feature mask

    @property
    def n_samples(self) -> int:
        return self.arr.shape[0] if self.valid is None else self.valid.size


def _align_y(y: xr.DataArray, batch: _Batch) -> xr.DataArray:
    """Put ``y``'s samples in the same order as ``X``'s, or raise."""
    sample_dim = batch.sample_dim
    if sample_dim not in y.dims:
        raise ValueError(f"y is missing sample_dim={sample_dim!r} (has dims={y.dims}).")
    n_y = y.sizes[sample_dim]
    if n_y != batch.n_samples:
        raise ValueError(
            f"y has {n_y} samples along {sample_dim!r} but X has {batch.n_samples}."
        )
    expected = batch.sample_index
    if expected is None or sample_dim not in y.indexes:
        return y  # nothing to align on — positional, sizes already agree
    current = y.indexes[sample_dim]
    if current.equals(expected):
        return y
    if current.is_unique and current.sort_values().equals(expected.sort_values()):
        return y.reindex({sample_dim: expected})
    raise ValueError(
        f"y's {sample_dim!r} coordinate does not match X's; align them first "
        "(e.g. xr.align(X, y, join='inner'))."
    )


def _marshal_y(
    y: xr.DataArray | xr.Dataset, batch: _Batch
) -> tuple[np.ndarray, Layout]:
    """Align an xarray target to ``X`` and flatten it.

    Returns:
        ``(values, layout)`` — ``values`` is ``(n_samples,)`` for a 1-D
        DataArray target and ``(n_samples, k)`` otherwise; ``layout`` is
        the target's grid, used to label ``predict`` outputs.
    """
    sample_dim = batch.sample_dim
    assert sample_dim is not None
    if isinstance(y, xr.Dataset):
        aligned = xr.Dataset(
            {name: _align_y(y[name], batch) for name in y.data_vars}, attrs=y.attrs
        )
        layout: Layout = dataset_layout(aligned, sample_dim)
        return to_2d(conform_dataset(aligned, layout, where="y"), layout), layout
    aligned = _align_y(y, batch)
    layout = array_layout(aligned, sample_dim)
    values = to_2d(aligned.transpose(sample_dim, *layout.feature_dims), layout)
    return (values if layout.feature_dims else values[:, 0]), layout


def _keeps_features(estimator: Any, n_in: int, out: np.ndarray) -> bool:
    """Whether a ``transform`` output column ``i`` is input feature ``i``.

    sklearn's feature-name API answers this exactly: one-to-one
    transformers (scalers, imputers that drop nothing, …) report their
    input names ``x0 … x{n-1}``, while reducers report their own
    (``pca0``, ``kmeans0``, …) — even when ``n_components == n_features``.
    Estimators without the API fall back to comparing column counts.
    """
    if out.ndim != 2 or out.shape[1] != n_in:
        return False
    try:
        names = estimator.get_feature_names_out()
    except Exception:  # no feature-name support — best effort by shape
        return True
    return [str(n) for n in names] == [f"x{i}" for i in range(n_in)]


def _estimator_has(*names: str) -> Callable[[XarrayEstimator], bool]:
    """``available_if`` check: does the delegate (fitted, else unfitted) have it?"""

    def check(self: XarrayEstimator) -> bool:
        est = self.__dict__.get("estimator_", self.estimator)
        return any(hasattr(est, name) for name in names)

    return check


def _explicit(
    sample_dim: Hashable | None,
    new_feature_dim: str | None,
    nan_policy: NanPolicy | None,
    missing: MissingKind | None = None,
    tree_mode: TreeMode | None = None,
    tree_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The wrapper settings a caller actually passed (``None`` = not passed)."""
    given = {
        "sample_dim": sample_dim,
        "new_feature_dim": new_feature_dim,
        "nan_policy": nan_policy,
        "missing": missing,
        "tree_mode": tree_mode,
        "tree_paths": None if tree_paths is None else list(tree_paths),
    }
    return {key: value for key, value in given.items() if value is not None}


def _check_no_conflict(
    wrap: XarrayEstimator,
    overrides: dict[str, Any],
) -> None:
    """Raise if ``overrides`` disagree with a fitted wrapper's own settings."""
    params = wrap.get_params(deep=False)
    if params["sample_dim"] is None and "sample_dim_" in wrap.__dict__:
        # Fitted with an inferred sample dim: restating it is not a conflict.
        params["sample_dim"] = wrap.__dict__["sample_dim_"]
    clashes = {k: v for k, v in overrides.items() if params[k] != v}
    if clashes:
        detail = ", ".join(
            f"{k}={v!r} (wrapper has {params[k]!r})" for k, v in clashes.items()
        )
        raise ValueError(
            f"Cannot override settings of a fitted XarrayEstimator: {detail}. "
            "Configure the wrapper before fitting instead."
        )


class XarrayEstimator(BaseEstimator):
    """Wrap an sklearn estimator so it operates on xarray inputs.

    The estimator is cloned on :meth:`fit` (so the original is never
    mutated) and stored as :attr:`estimator_`. After fitting, attribute
    access on the wrapper transparently proxies to ``estimator_`` —
    that means ``wrap.components_``, ``wrap.cluster_centers_``,
    ``wrap.coef_``, etc. work as you would expect. Methods the wrapped
    estimator lacks (``predict_proba`` on a scaler, say) are absent on
    the wrapper too, so ``hasattr`` duck-typing behaves as in sklearn.

    ``fit`` records the input's feature grid (feature dims, sizes and
    coordinates). Every later input to ``transform`` / ``predict`` /
    ``predict_proba`` / ``score`` must carry the same grid: feature dims
    may come in any order and indexed coordinates may be permuted — both
    are reordered to the fit-time layout — but a different grid (other
    dims, sizes, or coordinate labels) raises instead of being fed to the
    estimator column-for-column.

    Args:
        estimator: Any unfitted sklearn ``BaseEstimator``.
        sample_dim: The xarray dimension indexing samples. If ``None``,
            ``fit`` uses the first dim of its input and records it as
            ``sample_dim_`` for every later call.
        new_feature_dim: Name of the feature dimension when the
            estimator changes the number of features (e.g. PCA reducing
            150 features to 5 components).
        nan_policy: How missing values are handled around the estimator
            (see :mod:`xrsklearn._src.nan` for the full semantics):

            - ``"propagate"`` (default): hand them to the estimator.
            - ``"raise"``: raise if ``X`` (or ``y``) holds any.
            - ``"mask_features"``: drop feature columns that are missing in
              *every* fit sample — a land mask — and reuse that column
              mask at transform time; feature-space outputs get the
              columns back as missing.
            - ``"mask_samples"``: drop sample rows with any missing value
              in ``X`` (or ``y``, when fitting / scoring), along with the
              matching rows of per-sample fit arguments such as
              ``sample_weight``; every output gets the rows back as
              missing.
            - ``"mask"``: ``"mask_features"`` then ``"mask_samples"`` —
              the right choice for land-masked fields that also have gaps.

            For Dataset input the masks span the column-concatenation of
            all data variables. Refilled integer outputs (cluster labels)
            are promoted to ``float64``; string labels to ``object``.
        missing: What counts as missing: ``"nan"`` (default: NaN, NaT,
            ``None``) or ``"nonfinite"`` (also ±inf).
        tree_mode: What fitting on an ``xr.DataTree`` means (see
            :mod:`xrsklearn._src.tree`):

            - ``"per_node"`` (default): one estimator per data node, stored
              in ``estimators_``; nodes may have different grids.
            - ``"pool_samples"``: one estimator fitted on every node's
              samples stacked along ``sample_dim``; nodes share one grid.
            - ``"concat_features"``: one estimator whose features are every
              node's variables side by side; nodes share the sample axis.
        tree_paths: DataTree nodes to use; by default every node with data
            variables.

    Attributes:
        estimator_: The fitted clone of ``estimator``.
        sample_dim_: The sample dimension resolved at fit time (``None``
            when fitted on a NumPy array).
        layout_: The fit-time feature layout (``None`` when fitted on a
            NumPy array).
        target_layout_: The fit-time layout of an xarray ``y`` (``None``
            otherwise); ``predict`` outputs are rebuilt on it.
        estimators_: ``{path: fitted XarrayEstimator}`` after a
            ``tree_mode="per_node"`` fit on a DataTree (there is no
            ``estimator_`` then).
        tree_paths_: The DataTree node paths used at fit time.
        feature_mask_: Boolean keep-mask over the input feature columns
            learned by the ``"mask"`` / ``"mask_features"`` policies
            (``None`` when no column was dropped).

    Example:
        Decompose a (time, lat, lon) cube with PCA, recover the original
        grid, and reach into the fitted estimator's attributes:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.decomposition import PCA
        >>> rng = np.random.default_rng(0)
        >>> da = xr.DataArray(
        ...     rng.normal(size=(8, 3, 4)), dims=("time", "lat", "lon"),
        ... )
        >>> wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
        >>> scores = wrap.fit_transform(da)
        >>> scores.dims, scores.shape
        (('time', 'component'), (8, 3))
        >>> recon = wrap.inverse_transform(scores)
        >>> recon.dims, recon.shape
        (('time', 'lat', 'lon'), (8, 3, 4))
        >>> wrap.components_.shape  # passthrough to the fitted estimator
        (3, 12)

        ```

        Inputs are matched to the fit-time grid, so a transposed cube
        gives the same scores instead of silently permuted features:

        ```pycon
        >>> same = wrap.transform(da.transpose("time", "lon", "lat"))
        >>> bool(np.allclose(same, scores))
        True

        ```

        Cluster a multi-variable Dataset (data_vars column-concatenated
        into one feature matrix):

        ```pycon
        >>> from sklearn.cluster import KMeans
        >>> ds = xr.Dataset({"u": da, "v": da})
        >>> wrap = XarrayEstimator(
        ...     KMeans(n_clusters=2, n_init="auto", random_state=0),
        ...     sample_dim="time",
        ... )
        >>> labels = wrap.fit(ds).predict(ds)
        >>> labels.dims, labels.shape
        (('time',), (8,))
        >>> wrap.cluster_centers_.shape  # (n_clusters, n_concat_features)
        (2, 24)

        ```

        Land-masked field with a gap — ``nan_policy="mask"`` drops the
        always-missing land column and the incomplete sample row, then
        puts both back as NaN on the way out:

        ```pycon
        >>> ssh = da.copy()
        >>> ssh[:, 0, 0] = np.nan  # land, in every sample
        >>> ssh[3, 1, 1] = np.nan  # a gap in one sample
        >>> wrap = XarrayEstimator(
        ...     PCA(n_components=2), sample_dim="time", nan_policy="mask",
        ... )
        >>> scores = wrap.fit_transform(ssh)
        >>> int(wrap.feature_mask_.sum())  # 12 cells minus 1 land cell
        11
        >>> bool(scores[3].isnull().all()), int(scores.notnull().all("component").sum())
        (True, 7)
        >>> recon = wrap.inverse_transform(scores)
        >>> bool(recon[:, 0, 0].isnull().all())  # land comes back as NaN
        True

        ```
    """

    # Fitted state, set by ``fit``; declared here for type checkers.
    estimator_: Any
    estimators_: dict[str, XarrayEstimator]
    tree_paths_: tuple[str, ...]

    def __init__(
        self,
        estimator: BaseEstimator,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
        missing: MissingKind = "nan",
        tree_mode: TreeMode = "per_node",
        tree_paths: Sequence[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.sample_dim = sample_dim
        self.new_feature_dim = new_feature_dim
        self.nan_policy = nan_policy
        self.missing = missing
        self.tree_mode = tree_mode
        self.tree_paths = tree_paths

    # ---------- internals -------------------------------------------------

    def _check_params(self) -> None:
        """Validate constructor parameters (sklearn defers this to call time)."""
        check_policy(self.nan_policy, self.missing)
        modes = get_args(TreeMode)
        if self.tree_mode not in modes:
            raise ValueError(
                f"tree_mode must be one of {modes}; got {self.tree_mode!r}."
            )
        if not isinstance(self.new_feature_dim, str):
            raise TypeError(
                f"new_feature_dim must be a str; got {type(self.new_feature_dim)}."
            )
        if not hasattr(self.estimator, "fit") and "estimator_" not in self.__dict__:
            raise TypeError(
                f"estimator must implement fit(); got {type(self.estimator).__name__}."
            )

    def _resolve_sample_dim(
        self, x: xr.DataArray | xr.Dataset | dict[str, xr.Dataset]
    ) -> Hashable:
        fitted = self.__dict__.get("sample_dim_")
        if fitted is not None:
            return fitted
        if self.sample_dim is not None:
            return self.sample_dim
        if isinstance(x, dict):  # DataTree nodes: the first node decides
            x = next(iter(x.values()))
        if isinstance(x, xr.DataArray):
            return x.dims[0]
        # Dataset: use the first dim of the first variable.
        first = next(iter(x.data_vars))
        return x[first].dims[0]

    def _stack(
        self,
        x: XInput,
        *,
        space: Literal["fit", "features", "output"],
    ) -> _Batch:
        """Marshal ``x`` into a 2-D array and apply the feature-side NaN policy.

        Args:
            x: The input.
            space: Which space ``x`` lives in. ``"fit"``: the training
                input — its layout and feature mask are learned.
                ``"features"``: the training feature space — ``x`` is
                conformed to the fit-time layout and the fit-time feature
                mask is applied. ``"output"``: the estimator's output space
                (``inverse_transform`` of PCA scores, say) — neither applies.

        Returns:
            The marshalled batch. NumPy inputs pass through unlabeled.
        """
        self._check_params()
        if space != "fit" and "estimators_" in self.__dict__:
            raise TypeError(
                f"{type(self).__name__} was fit per DataTree node "
                f"({', '.join(self.__dict__['estimators_'])}); pass a DataTree, "
                "or use .estimators_[path] for a single node."
            )
        if isinstance(x, np.ndarray):
            batch = _Batch(arr=x)
            self._apply_feature_policy(batch, space)
            return batch
        if isinstance(x, xr.DataTree):
            batch = self._stack_tree(x, space=space)
            self._apply_feature_policy(batch, space)
            return batch
        if not isinstance(x, xr.DataArray | xr.Dataset):
            raise TypeError(
                "X must be xr.DataArray, xr.Dataset, xr.DataTree, or np.ndarray; "
                f"got {type(x)}."
            )
        sample_dim = self._resolve_sample_dim(x)
        fitted = self.__dict__.get("layout_") if space == "features" else None
        if isinstance(fitted, TreeLayout):
            raise TypeError("Estimator was fit on a DataTree; got " + type(x).__name__)
        layout: Layout
        if isinstance(x, xr.DataArray):
            in_dims: Any = tuple(x.dims)
            if isinstance(fitted, DatasetLayout):
                raise TypeError("Estimator was fit on a Dataset; got a DataArray.")
            if isinstance(fitted, ArrayLayout):
                x, layout = conform_array(x, fitted), fitted
            else:
                layout = array_layout(x, sample_dim)
                x = x.transpose(sample_dim, *layout.feature_dims)
        else:
            in_dims = {name: tuple(var.dims) for name, var in x.data_vars.items()}
            if isinstance(fitted, ArrayLayout):
                raise TypeError("Estimator was fit on a DataArray; got a Dataset.")
            if isinstance(fitted, DatasetLayout):
                x, layout = conform_dataset(x, fitted), fitted
            else:
                layout = dataset_layout(x, sample_dim)
                x = conform_dataset(x, layout)
        batch = _Batch(
            arr=to_2d(x, layout),
            layout=layout,
            sample_dim=sample_dim,
            samples=sample_coords(x, sample_dim),
            sample_index=x.indexes.get(sample_dim),
            in_dims=in_dims,
            name=x.name if isinstance(x, xr.DataArray) else None,
            attrs=dict(x.attrs) if isinstance(x, xr.DataArray) else {},
        )
        self._apply_feature_policy(batch, space)
        return batch

    def _stack_tree(self, tree: xr.DataTree, *, space: str) -> _Batch:
        """Marshal DataTree nodes side by side (``tree_mode="concat_features"``)."""
        if self.tree_mode != "concat_features":
            raise TypeError(  # other modes dispatch before marshalling
                f"tree_mode={self.tree_mode!r} does not marshal a whole DataTree."
            )
        nodes = select_nodes(tree, self.__dict__.get("tree_paths_", self.tree_paths))
        sample_dim = self._resolve_sample_dim(nodes)
        fitted = self.__dict__.get("layout_") if space == "features" else None
        if fitted is not None and not isinstance(fitted, TreeLayout):
            raise TypeError("Estimator was not fit on a DataTree; got a DataTree.")
        layout = fitted if fitted is not None else tree_layout(nodes, sample_dim)
        nodes = conform_tree(nodes, layout)
        first = nodes[layout.paths[0]]
        return _Batch(
            arr=to_2d(nodes, layout),
            layout=layout,
            sample_dim=sample_dim,
            samples=sample_coords(first, sample_dim),
            sample_index=first.indexes.get(sample_dim),
            in_dims={
                path: {name: tuple(var.dims) for name, var in ds.data_vars.items()}
                for path, ds in nodes.items()
            },
            attrs={},
        )

    def _tree_dispatch(self, x: Any) -> TypeGuard[xr.DataTree]:
        """Whether ``x`` is a DataTree handled node-wise (not column-joined)."""
        return isinstance(x, xr.DataTree) and self.tree_mode != "concat_features"

    def _apply_feature_policy(self, batch: _Batch, space: str) -> None:
        """``raise`` check on X, then drop fit-time all-missing columns."""
        if self.nan_policy == "raise":
            check_no_missing(batch.arr, label="X", missing=self.missing)
        if not masks_features(self.nan_policy) or space == "output":
            return
        if space == "fit":
            batch.full = batch.arr
            keep = learn_feature_mask(batch.arr, self.missing)
        else:
            if "feature_mask_" not in self.__dict__:
                # A raw fitted estimator (accessor / SklearnOp) never saw the
                # fit-time columns, so there is no mask to reapply.
                raise ValueError(
                    f"nan_policy={self.nan_policy!r} needs the feature mask "
                    "learned at fit time; fit an XarrayEstimator with this "
                    "nan_policy and pass it instead of a raw estimator."
                )
            keep = self.__dict__.get("feature_mask_")
            if keep is not None and keep.size != batch.arr.shape[1]:
                raise ValueError(
                    f"X has {batch.arr.shape[1]} features but the fit-time "
                    f"feature mask covers {keep.size}."
                )
        if keep is not None:
            batch.arr = batch.arr[:, keep]
            batch.features = keep

    def _refine_feature_mask(self, batch: _Batch, y: np.ndarray | None) -> None:
        """Relearn the fit-time feature mask from rows whose target is present.

        Rows dropped for a missing ``y`` must not keep an otherwise empty
        column alive, or sample masking would then reject every row.
        """
        if y is None or batch.full is None or not masks_samples(self.nan_policy):
            return
        usable = ~_rows(missing_mask(np.asarray(y), self.missing)).any(axis=1)
        if usable.all() or not usable.any():
            return
        keep = learn_feature_mask(batch.full[usable], self.missing)
        batch.arr = batch.full if keep is None else batch.full[:, keep]
        batch.features = keep

    def _mask_samples(
        self,
        batch: _Batch,
        y: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Sample-side NaN policy: check or drop incomplete rows of X and y.

        Returns:
            ``y`` and ``params`` subset to the kept rows. ``batch`` is
            updated in place (``arr`` subset, ``valid`` set).
        """
        params = {} if params is None else params
        if self.nan_policy == "raise" and y is not None:
            check_no_missing(np.asarray(y), label="y", missing=self.missing)
        if not masks_samples(self.nan_policy):
            return y, params
        keep = learn_sample_mask(batch.arr, y, missing=self.missing)
        if keep is None:
            return y, params
        batch.arr = batch.arr[keep]
        batch.valid = keep
        return (None if y is None else y[keep]), mask_fit_params(params, keep)

    def _prepare_fit_params(
        self, params: dict[str, Any], batch: _Batch
    ) -> dict[str, Any]:
        """Align xarray fit arguments (e.g. ``sample_weight``) to X's samples."""
        out = dict(params)
        for key, value in params.items():
            if isinstance(value, xr.DataArray | xr.Dataset):
                if batch.sample_dim is None:
                    raise TypeError(
                        f"Fit argument {key!r} is an xarray object but X is a "
                        "NumPy array; pass NumPy instead."
                    )
                out[key] = _marshal_y(value, batch)[0]
        return out

    def _prepare_y(
        self,
        y: YInput,
        batch: _Batch,
    ) -> tuple[np.ndarray | None, Layout | None]:
        """Marshal ``y`` to numpy, aligned to and masked like ``X``.

        Returns:
            ``(values, layout)`` — ``layout`` is the target's grid when
            ``y`` is an xarray object, else ``None``.

        Raises:
            TypeError: If ``X`` was NumPy but ``y`` is an xarray object.
            ValueError: If an xarray ``y`` cannot be aligned to ``X``'s
                sample axis.
        """
        if y is None:
            return None, None
        if isinstance(y, xr.DataTree) or (
            isinstance(y, Mapping) and not isinstance(y, xr.Dataset)
        ):
            raise TypeError(
                "A DataTree / mapping y is per node; it needs a DataTree X with "
                "tree_mode='per_node' or 'pool_samples'."
            )
        layout: Layout | None = None
        if isinstance(y, xr.DataArray | xr.Dataset):
            if batch.sample_dim is None:
                raise TypeError(
                    "When x is a NumPy array, y must also be a NumPy array or "
                    "None; xarray y requires a sample dimension carried on x."
                )
            y_np, layout = _marshal_y(y, batch)
        else:
            y_np = np.asarray(y)
        return y_np, layout

    # ---------- output labeling -------------------------------------------
    #
    # Each verb knows which space its output lives in, so the layout is
    # chosen per method rather than guessed from the column count:
    #
    #   transform          input grid if one-to-one (``_keeps_features``),
    #                      else (sample_dim, new_feature_dim)
    #   inverse_transform  the fit-time input grid
    #   predict            the fit-time *target* grid (from an xarray ``y``)
    #   predict_proba      (sample_dim, "class") labeled by ``classes_``

    def _features(
        self,
        out: np.ndarray,
        batch: _Batch,
        layout: Layout,
        dims: Any = None,
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Label an output that lives on ``layout``'s feature grid."""
        assert batch.samples is not None
        if isinstance(layout, TreeLayout):
            return tree_from_2d(
                out,
                layout,
                batch.samples,
                dims=dims if isinstance(dims, dict) else None,
            )
        if isinstance(layout, DatasetLayout):
            return dataset_from_2d(
                out,
                layout,
                batch.samples,
                dims=dims if isinstance(dims, dict) else None,
            )
        return grid_from_2d(
            out, layout, batch.samples, dims=dims if isinstance(dims, tuple) else None
        )

    def _generic(
        self,
        out: np.ndarray,
        batch: _Batch,
        *,
        feature_dim: Hashable | None = None,
        feature_coord: np.ndarray | None = None,
        name: Hashable | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Label an output that left the input feature grid."""
        assert batch.sample_dim is not None and batch.samples is not None
        return generic_from_2d(
            out,
            batch.sample_dim,
            batch.samples,
            new_feature_dim=self.new_feature_dim
            if feature_dim is None
            else feature_dim,
            feature_coord=feature_coord,
            name=name,
            attrs=attrs,
        )

    def _label_transform(self, out: Any, batch: _Batch) -> Output:
        out = np.asarray(out)
        keeps = _keeps_features(self.estimator_, batch.arr.shape[1], out)
        out = restore_rows(out, batch.valid)
        if keeps:
            # Masked-out feature columns come back as missing on the grid.
            out = restore_columns(out, batch.features)
        if batch.layout is None:
            return out
        if keeps:
            result = self._features(out, batch, batch.layout, batch.in_dims)
            if isinstance(result, xr.DataArray):
                # batch.layout is the fit-time grid; the metadata is the input's.
                result = result.rename(batch.name)
                result.attrs = dict(batch.attrs)
            return result
        # Reduced output of a DataArray keeps its identity (name, attrs);
        # for a Dataset there is no single variable to inherit them from.
        return self._generic(out, batch, name=batch.name, attrs=batch.attrs)

    def _label_inverse(self, out: Any, batch: _Batch) -> Output:
        out = restore_rows(np.asarray(out), batch.valid)
        keep = self.__dict__.get("feature_mask_")
        if keep is not None and out.ndim == 2 and out.shape[1] == keep.sum():
            out = restore_columns(out, keep)
        train = self.__dict__.get("layout_")
        if batch.layout is None:
            return out
        if train is not None and out.ndim == 2 and out.shape[1] == train.n_features:
            # Feature grid from training, sample axis from the input — which
            # may cover a different period than the training set.
            return self._features(out, batch, train)
        return self._generic(out, batch, name=batch.name, attrs=batch.attrs)

    def _label_target(self, out: Any, batch: _Batch) -> Output:
        out = restore_rows(np.asarray(out), batch.valid)
        if batch.layout is None:
            return out
        target = self.__dict__.get("target_layout_")
        columns = out[:, None] if out.ndim == 1 else out
        if target is not None and columns.shape[1] == target.n_features:
            return self._features(columns, batch, target)
        # No labeled target (unsupervised ``predict``, or NumPy ``y``): the
        # result is not in the input's units, so it inherits no attrs.
        return self._generic(out, batch)

    def _label_proba(
        self, out: Any, batch: _Batch, classes: Any
    ) -> xr.DataArray | np.ndarray:
        out = restore_rows(np.asarray(out), batch.valid)
        if batch.layout is None:
            return out
        coord = None
        if classes is not None and len(classes) == out.shape[1]:
            coord = np.asarray(classes)
        if batch.sample_dim == CLASS_DIM:
            raise ValueError(
                f"predict_proba labels classes along a {CLASS_DIM!r} dim, which "
                f"clashes with sample_dim={CLASS_DIM!r}; rename the sample "
                "dimension first."
            )
        return self._generic(out, batch, feature_dim=CLASS_DIM, feature_coord=coord)

    def _record_fit(self, batch: _Batch, target: Layout | None) -> None:
        self.sample_dim_ = batch.sample_dim
        self.layout_ = batch.layout
        # Only supervised predictions live in y-space; an unsupervised
        # estimator (KMeans) accepts and ignores y, so its labels must not
        # inherit the target's name, attrs or grid.
        supervised = is_classifier(self.estimator_) or is_regressor(self.estimator_)
        self.target_layout_ = target if supervised else None
        self.feature_mask_ = batch.features
        if isinstance(batch.layout, TreeLayout):
            self.tree_paths_ = batch.layout.paths

    # ---------- sklearn-style verbs ---------------------------------------

    def fit(
        self,
        x: XInput,
        y: YInput = None,
        **kwargs: Any,
    ) -> XarrayEstimator:
        """Fit the wrapped estimator to ``x`` (and optional ``y``)."""
        self._reset()
        if self._tree_dispatch(x):
            self._check_params()
            fit_tree = fit_per_node if self.tree_mode == "per_node" else fit_pooled
            fit_tree(self, x, y, kwargs, transform=False)
            return self
        batch = self._stack(x, space="fit")
        y_np, target = self._prepare_y(y, batch)
        params = self._prepare_fit_params(kwargs, batch)
        self._refine_feature_mask(batch, y_np)
        y_np, params = self._mask_samples(batch, y_np, params)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(batch.arr, y_np, **params)
        self._record_fit(batch, target)
        return self

    @available_if(_estimator_has("transform"))
    def transform(self, x: XInput) -> Output:
        """Transform ``x`` via the fitted estimator.

        One-to-one transformers (scalers, …) return data on the input's
        grid — a Dataset for Dataset input. Anything else (PCA scores,
        KMeans distances) returns ``(sample_dim, new_feature_dim)``.
        """
        self._require_fitted()
        if self._tree_dispatch(x):
            return self._apply_tree("transform", x)
        batch = self._stack(x, space="features")
        self._mask_samples(batch)
        return self._label_transform(self.estimator_.transform(batch.arr), batch)

    @available_if(_estimator_has("fit_transform", "transform"))
    def fit_transform(
        self,
        x: XInput,
        y: YInput = None,
        **kwargs: Any,
    ) -> Output:
        """Fit then transform ``x`` (output layout as in :meth:`transform`)."""
        self._reset()
        if self._tree_dispatch(x):
            self._check_params()
            fit_tree = fit_per_node if self.tree_mode == "per_node" else fit_pooled
            result = fit_tree(self, x, y, kwargs, transform=True)
            assert result is not None  # transform=True always returns the tree
            return result
        batch = self._stack(x, space="fit")
        y_np, target = self._prepare_y(y, batch)
        params = self._prepare_fit_params(kwargs, batch)
        self._refine_feature_mask(batch, y_np)
        y_np, params = self._mask_samples(batch, y_np, params)
        self.estimator_ = clone(self.estimator)
        if hasattr(self.estimator_, "fit_transform"):
            out = self.estimator_.fit_transform(batch.arr, y_np, **params)
        else:
            self.estimator_.fit(batch.arr, y_np, **params)
            out = self.estimator_.transform(batch.arr)
        self._record_fit(batch, target)
        return self._label_transform(out, batch)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, x: XInput) -> Output:
        """Map back to the original feature space via the fitted estimator.

        The result is rebuilt on the fit-time grid — a Dataset if the
        estimator was fit on one — with feature coordinates from training
        and sample coordinates from ``x`` (which may cover a different
        period than the training set). ``x`` may be component scores
        (PCA) or grid-shaped data (a scaler's output); the latter is
        checked against the fit-time layout like any other input.
        """
        self._require_fitted()
        if self._tree_dispatch(x):
            return self._apply_tree("inverse_transform", x)
        train = self.__dict__.get("layout_")
        keep = self.__dict__.get("feature_mask_")
        in_feature_space = (
            train is not None
            and isinstance(x, xr.DataArray | xr.Dataset | xr.DataTree)
            and matches(x, train)
        ) or (
            # A full-width NumPy array (e.g. a one-to-one fit_transform output
            # with the masked columns restored) is in feature space too.
            isinstance(x, np.ndarray)
            and keep is not None
            and x.ndim == 2
            and x.shape[1] == keep.size
        )
        batch = self._stack(x, space="features" if in_feature_space else "output")
        self._mask_samples(batch)
        return self._label_inverse(self.estimator_.inverse_transform(batch.arr), batch)

    @available_if(_estimator_has("predict"))
    def predict(self, x: XInput) -> Output:
        """Predict via the fitted estimator (regression / classification).

        If ``fit`` received an xarray ``y``, predictions come back on its
        grid, with its name and attrs (a Dataset for a Dataset target).
        Otherwise they are ``(sample_dim,)`` or
        ``(sample_dim, new_feature_dim)`` with no attrs.
        """
        self._require_fitted()
        if self._tree_dispatch(x):
            return self._apply_tree("predict", x)
        batch = self._stack(x, space="features")
        self._mask_samples(batch)
        return self._label_target(self.estimator_.predict(batch.arr), batch)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, x: XInput) -> Output | list[xr.DataArray | np.ndarray]:
        """Class-probability prediction (classifiers only).

        Returns ``(sample_dim, "class")`` with the ``class`` coordinate set
        to the estimator's ``classes_``; a list of such arrays for
        multi-output classifiers.
        """
        self._require_fitted()
        if self._tree_dispatch(x):
            return self._apply_tree("predict_proba", x)
        batch = self._stack(x, space="features")
        self._mask_samples(batch)
        out = self.estimator_.predict_proba(batch.arr)
        classes = getattr(self.estimator_, "classes_", None)
        if isinstance(out, list):
            return [
                self._label_proba(o, batch, None if classes is None else classes[i])
                for i, o in enumerate(out)
            ]
        return self._label_proba(out, batch, classes)

    @available_if(_estimator_has("score"))
    def score(
        self,
        x: XInput,
        y: YInput = None,
    ) -> float | dict[str, float]:
        """Scalar score from the wrapped estimator.

        Not re-wrapped — sklearn ``.score`` returns a Python float.
        """
        self._require_fitted()
        if self._tree_dispatch(x):
            if self.tree_mode == "per_node":
                return score_per_node(self, x, y)
            return score_pooled(self, x, y)
        batch = self._stack(x, space="features")
        y_np, _ = self._prepare_y(y, batch)
        y_np, _ = self._mask_samples(batch, y_np)
        return float(self.estimator_.score(batch.arr, y_np))

    # ---------- proxy + dunder --------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only invoked if the attribute is not found the normal way.
        # Forward fitted-state attributes (``components_``, ``coef_``,
        # ``cluster_centers_``, ``n_iter_`` …) to the wrapped estimator.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _GATED_VERBS:
            # ``available_if`` hid the method because the delegate lacks it.
            est = self.__dict__.get("estimator_", self.estimator)
            raise AttributeError(
                f"{type(est).__name__} does not implement {name}, so neither "
                f"does this {type(self).__name__}."
            )
        if "estimators_" in self.__dict__ and not name.endswith("__"):
            raise AttributeError(
                f"{type(self).__name__} was fit per DataTree node; use "
                f".estimators_[path].{name} for a node's fitted attribute."
            )
        try:
            est = self.__dict__["estimator_"]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r} "
                "(estimator has not been fitted yet)."
            ) from exc
        return getattr(est, name)

    def _reset(self) -> None:
        """Forget fitted state so a refit starts clean."""
        for name in ("sample_dim_", "estimators_", "tree_paths_", "estimator_"):
            self.__dict__.pop(name, None)

    def _apply_tree(self, method: str, tree: xr.DataTree) -> xr.DataTree:
        if "estimators_" in self.__dict__:
            return apply_per_node(self, method, tree)
        if self.tree_mode == "per_node":
            raise TypeError(
                "This estimator was fit on a single DataArray / Dataset; with "
                "tree_mode='per_node' it can only be applied to a DataTree it was "
                "fit on. Use tree_mode='pool_samples' to apply one estimator to "
                "every node."
            )
        return apply_pooled(self, method, tree)

    def _require_fitted(self) -> None:
        if "estimator_" not in self.__dict__ and "estimators_" not in self.__dict__:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted; call .fit(...) first."
            )

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        return (
            f"XarrayEstimator(estimator={self.estimator!r}, "
            f"sample_dim={self.sample_dim!r}, "
            f"new_feature_dim={self.new_feature_dim!r}, "
            f"nan_policy={self.nan_policy!r}, "
            f"missing={self.missing!r}, "
            f"tree_mode={self.tree_mode!r}, "
            f"tree_paths={self.tree_paths!r})"
        )
