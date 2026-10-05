"""Missing-value handling around an sklearn delegate.

Most sklearn estimators reject NaN, while gridded Earth-system data is full
of it. It shows up in two shapes, and each needs its own treatment:

- **Missing features** — the same grid cells are empty in every sample. A
  land mask on an ocean field is the canonical case: every time step has
  NaN over land. Dropping *samples* with any NaN would drop every sample,
  so these columns are dropped instead. Which columns to drop is learned
  once, at fit time, and reused for every later call so the estimator
  always sees the same columns.
- **Missing samples** — scattered gaps (cloud cover, sensor dropouts)
  that make individual rows incomplete. Those rows are dropped before the
  estimator sees them, together with the matching entries of ``y`` and of
  per-sample fit arguments such as ``sample_weight``, and reinserted as
  missing on the way out.

The policies:

============== =========================================================
``propagate``  hand everything to the estimator unchanged (default)
``raise``      raise before the estimator sees any missing value
``mask_features`` drop feature columns that are missing in *every* fit
               sample; the same columns are dropped at transform time
               and restored as missing in feature-space outputs
``mask_samples`` drop sample rows with any missing value in ``X`` (or in
               ``y`` when fitting / scoring) and restore them as missing
               rows in every output
``mask``       ``mask_features`` then ``mask_samples`` — the right choice
               for land-masked fields that also have gaps
============== =========================================================

``missing="nan"`` (default) treats NaN / NaT / ``None`` as missing;
``missing="nonfinite"`` also treats ±inf as missing.

Masked rows and columns are refilled with a missing value of a dtype that
can hold it: floats keep their dtype, integers and bools are promoted to
``float64``, datetimes use ``NaT``, and strings / objects become ``object``
arrays holding ``NaN`` (never the string ``"nan"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from sklearn.utils.validation import _check_method_params


#: How missing values are handled around the estimator (see the module docstring).
NanPolicy = Literal["propagate", "raise", "mask", "mask_samples", "mask_features"]
#: What counts as missing: ``"nan"`` (NaN / NaT / ``None``) or ``"nonfinite"`` (+ ±inf).
MissingKind = Literal["nan", "nonfinite"]


def check_policy(policy: str, missing: str) -> None:
    """Validate a ``(nan_policy, missing)`` pair.

    Args:
        policy: Candidate NaN policy.
        missing: Candidate missing-value kind.

    Raises:
        ValueError: If either value is not one of the allowed literals.
    """
    if policy not in get_args(NanPolicy):
        raise ValueError(
            f"nan_policy must be one of {get_args(NanPolicy)}; got {policy!r}."
        )
    if missing not in get_args(MissingKind):
        raise ValueError(
            f"missing must be one of {get_args(MissingKind)}; got {missing!r}."
        )


def masks_features(policy: NanPolicy) -> bool:
    """Whether ``policy`` drops all-missing feature columns."""
    return policy in ("mask", "mask_features")


def masks_samples(policy: NanPolicy) -> bool:
    """Whether ``policy`` drops sample rows containing missing values."""
    return policy in ("mask", "mask_samples")


def missing_mask(arr: np.ndarray, missing: MissingKind = "nan") -> np.ndarray:
    """Element-wise missing-value mask for an array of any dtype.

    Args:
        arr: Input array.
        missing: ``"nan"`` for NaN / NaT / ``None``; ``"nonfinite"`` to also
            count ±inf.

    Returns:
        A boolean array shaped like ``arr``. Integer, bool and string
        arrays cannot hold a missing value and yield all ``False``.
    """
    kind = arr.dtype.kind
    if kind in "fc":
        return ~np.isfinite(arr) if missing == "nonfinite" else np.isnan(arr)
    if kind in "mM":
        return np.isnat(arr)
    if kind == "O":
        mask = np.asarray(pd.isna(arr), dtype=bool)
        if missing == "nonfinite" and arr.size:
            mask |= np.vectorize(_is_inf, otypes=[bool])(arr)
        return mask
    return np.zeros(arr.shape, dtype=bool)


def _is_inf(value: Any) -> bool:
    try:
        return bool(np.isinf(value))
    except (TypeError, ValueError):
        return False


def _rows(arr: np.ndarray) -> np.ndarray:
    """View ``arr`` as ``(n_samples, -1)`` so row-wise reductions work."""
    return arr.reshape(arr.shape[0], -1)


def count_missing(arr: np.ndarray, missing: MissingKind = "nan") -> int:
    """Number of missing elements in ``arr``."""
    return int(missing_mask(arr, missing).sum())


def check_no_missing(arr: np.ndarray, *, label: str, missing: MissingKind) -> None:
    """Raise if ``arr`` holds any missing value (``nan_policy="raise"``).

    Args:
        arr: Array to check.
        label: What ``arr`` is, for the message (``"X"``, ``"y"``).
        missing: Missing-value kind.

    Raises:
        ValueError: If any element is missing.
    """
    n = count_missing(arr, missing)
    if n:
        what = "NaN/inf" if missing == "nonfinite" else "NaN"
        raise ValueError(
            f"{label} contains {n} missing ({what}) value(s). Use "
            "nan_policy='mask' to drop them around the estimator, "
            "'propagate' to forward them to it, or impute upstream."
        )


def learn_feature_mask(
    arr: np.ndarray, missing: MissingKind = "nan"
) -> np.ndarray | None:
    """Columns to keep: those not missing in *every* sample.

    Args:
        arr: ``(n_samples, n_features)`` fit-time matrix.
        missing: Missing-value kind.

    Returns:
        A boolean keep-mask over columns, or ``None`` when no column is
        entirely missing (nothing to drop).

    Raises:
        ValueError: If every column is entirely missing.
    """
    empty = missing_mask(arr, missing).all(axis=0)
    if not empty.any():
        return None
    if empty.all():
        raise ValueError(
            "Every feature column is missing in every sample; there is "
            "nothing left to fit."
        )
    return ~empty


def learn_sample_mask(
    arr: np.ndarray,
    *targets: np.ndarray | None,
    missing: MissingKind = "nan",
) -> np.ndarray | None:
    """Rows to keep: those with no missing value in ``arr`` or any target.

    Args:
        arr: ``(n_samples, n_features)`` matrix.
        *targets: Arrays aligned with ``arr``'s rows (``y``); ``None``
            entries are skipped.
        missing: Missing-value kind.

    Returns:
        A boolean keep-mask over rows, or ``None`` when no row has a
        missing value.

    Raises:
        ValueError: If every row has a missing value.
    """
    bad = missing_mask(arr, missing).any(axis=1)
    for target in targets:
        if target is not None:
            bad |= _rows(missing_mask(np.asarray(target), missing)).any(axis=1)
    if not bad.any():
        return None
    if bad.all():
        raise ValueError(
            "nan_policy removed all sample rows; at least one complete sample "
            "row is required before delegating to sklearn."
        )
    return ~bad


def fill_dtype(dtype: np.dtype) -> tuple[np.dtype, Any]:
    """The dtype and fill value used to reinsert masked entries.

    Args:
        dtype: Dtype of the estimator output.

    Returns:
        ``(dtype, fill)``: floats and complexes keep their dtype with NaN,
        datetimes / timedeltas keep theirs with NaT, integers and bools
        become ``float64`` with NaN, and anything else (strings, objects)
        becomes ``object`` with NaN.
    """
    kind = dtype.kind
    if kind in "fc":
        return dtype, np.nan
    if kind in "mM":
        return dtype, np.array("NaT", dtype=dtype)[()]
    if kind in "iub":
        return np.dtype(np.float64), np.nan
    return np.dtype(object), np.nan


def restore_rows(arr: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
    """Reinsert rows removed by a sample mask as missing values.

    Args:
        arr: Output computed on the kept rows.
        keep: Row keep-mask from :func:`learn_sample_mask` (``None`` = no-op).

    Returns:
        An array with ``keep.size`` rows.

    Raises:
        ValueError: If ``arr`` does not have one row per kept sample.
    """
    if keep is None:
        return arr
    n_kept = int(keep.sum())
    if arr.shape[0] != n_kept:
        raise ValueError(
            "Cannot restore masked sklearn output: output sample count "
            f"{arr.shape[0]} does not match valid input sample count {n_kept}."
        )
    dtype, fill = fill_dtype(arr.dtype)
    full = np.full((keep.size, *arr.shape[1:]), fill, dtype=dtype)
    full[keep] = arr
    return full


def restore_columns(arr: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
    """Reinsert columns removed by a feature mask as missing values.

    Args:
        arr: ``(n_samples, n_kept_features)`` output in feature space.
        keep: Column keep-mask from :func:`learn_feature_mask`.

    Returns:
        A ``(n_samples, keep.size)`` array.
    """
    if keep is None:
        return arr
    dtype, fill = fill_dtype(arr.dtype)
    full = np.full((arr.shape[0], keep.size), fill, dtype=dtype)
    full[:, keep] = arr
    return full


def mask_fit_params(
    params: Mapping[str, Any], keep: np.ndarray | None
) -> dict[str, Any]:
    """Drop masked rows from per-sample fit arguments (``sample_weight``, …).

    A parameter is treated as per-sample when it is array-like with one
    entry per input sample — the same rule sklearn itself uses to route
    fit parameters (``sklearn.utils.validation._check_method_params``).

    Args:
        params: Keyword arguments destined for the estimator's ``fit``.
        keep: Row keep-mask (``None`` = no-op).

    Returns:
        A new dict with per-sample arrays subset to the kept rows.
    """
    if keep is None:
        return dict(params)
    # Delegate to sklearn's own routing so sparse matrices, DataFrames and
    # other indexable containers are subset without being coerced.
    # Strings have a length but are never per-sample.
    routed = {k: v for k, v in params.items() if not isinstance(v, str | bytes)}
    shape_only = np.empty((keep.size, 0))
    subset = _check_method_params(shape_only, routed, indices=np.flatnonzero(keep))
    return {**params, **subset}
