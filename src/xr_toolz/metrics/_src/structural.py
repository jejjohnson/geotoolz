"""Structural metrics — SSIM, gradient/phase displacement.

V2.1. Diagnose geometry, gradient, and phase rather than pointwise
differences. Intended for cases where features are roughly correct
but slightly displaced — e.g. an SSH eddy 20 km off the reference
location.

Operators:

- :class:`SSIM` — structural similarity index via :mod:`scikit-image`.
- :class:`GradientDifference` — RMS difference of finite-difference
  gradients along ``dims``.
- :class:`PhaseShiftError` — best-aligned shift between prediction
  and reference via cross-correlation; supports periodic wrap.
- :class:`CentroidDisplacement` — distance between matched object
  centroids in two labelled object datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from skimage.metrics import structural_similarity

from xr_toolz.core import Operator


def _normalize_dims(dims: str | Sequence[str]) -> list[str]:
    return [dims] if isinstance(dims, str) else list(dims)


# ---------- SSIM ----------------------------------------------------------


def ssim(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    dims: str | Sequence[str],
    window: int | None = None,
) -> xr.DataArray:
    """Structural similarity index over ``dims`` per leading-dim slice.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable name.
        dims: One or two image dims (e.g. ``("lat", "lon")``).
        window: Optional :func:`skimage.metrics.structural_similarity`
            ``win_size`` override.

    Returns:
        Per-non-image-slice SSIM as a :class:`xr.DataArray` reduced
        over ``dims``.
    """
    core = _normalize_dims(dims)
    da_p = ds_pred[variable]
    da_r = ds_ref[variable]
    rest = [d for d in da_p.dims if d not in core]

    p = da_p.transpose(*rest, *core).values
    r = da_r.transpose(*rest, *core).values
    rest_shape = p.shape[: len(rest)]
    img_shape = p.shape[len(rest) :]
    p_flat = p.reshape(-1, *img_shape)
    r_flat = r.reshape(-1, *img_shape)

    base_kw: dict[str, Any] = {}
    if window is not None:
        base_kw["win_size"] = window

    def _ssim_slice(p_slc: np.ndarray, r_slc: np.ndarray) -> float:
        # Compute data_range per-slice; if the reference is constant or
        # all-NaN the standard formula collapses to 1.0 (perfect match
        # iff the prediction matches; otherwise no meaningful structural
        # comparison and we fall back to the equality test).
        ref_max = np.nanmax(r_slc)
        ref_min = np.nanmin(r_slc)
        rng = float(ref_max - ref_min) if np.isfinite(ref_max - ref_min) else 0.0
        if rng <= 0.0:
            return 1.0 if np.allclose(p_slc, r_slc, equal_nan=True) else 0.0
        return float(structural_similarity(p_slc, r_slc, data_range=rng, **base_kw))

    out = np.array([_ssim_slice(p_flat[i], r_flat[i]) for i in range(p_flat.shape[0])])
    out = out.reshape(rest_shape) if rest_shape else float(out.item())
    coords = {
        name: coord
        for name, coord in da_p.coords.items()
        if set(coord.dims).issubset(set(rest))
    }
    return xr.DataArray(np.asarray(out), dims=tuple(rest), coords=coords)


# ---------- Gradient difference ------------------------------------------


def gradient_difference(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    dims: str | Sequence[str],
) -> xr.DataArray:
    """RMS of the finite-difference gradient discrepancy.

    Computed as ``sqrt(<sum_d (∂_d pred - ∂_d ref)^2>)`` averaged
    over ``dims``. Constant offsets cancel; pure shifts contribute
    only at the boundary of the field of view.
    """
    core = _normalize_dims(dims)
    da_p = ds_pred[variable]
    da_r = ds_ref[variable]

    sq = xr.zeros_like(da_p, dtype=np.float64)
    for d in core:
        gp = da_p.differentiate(d)
        gr = da_r.differentiate(d)
        sq = sq + (gp - gr) ** 2
    return sq.mean(dim=core) ** 0.5


# ---------- Phase shift error --------------------------------------------


def phase_shift_error(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    *,
    variable: str,
    dims: str | Sequence[str],
    periodic: bool = False,
) -> xr.Dataset:
    """Estimate the integer-pixel shift that best aligns pred with ref.

    Uses the FFT-based cross-correlation peak. Returns one shift per
    image dim plus the residual RMSE after applying that shift.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable name.
        dims: Image dims (1-D or 2-D supported).
        periodic: When True, treat ``dims`` as periodic (longitude
            wrap). When False the search is still circular but the
            shift is reported in ``[-N/2, N/2)``; large shifts in
            non-periodic data should be interpreted with care.

    Returns:
        Dataset with one ``"shift_<dim>"`` variable per image dim and
        a ``"residual_rmse"`` variable.
    """
    core = _normalize_dims(dims)
    if len(core) not in (1, 2):
        raise ValueError("phase_shift_error supports 1-D or 2-D image dims only.")
    da_p = ds_pred[variable]
    da_r = ds_ref[variable]
    rest = [d for d in da_p.dims if d not in core]
    p = da_p.transpose(*rest, *core).values
    r = da_r.transpose(*rest, *core).values
    rest_shape = p.shape[: len(rest)]
    img_shape = p.shape[len(rest) :]
    p_flat = p.reshape(-1, *img_shape)
    r_flat = r.reshape(-1, *img_shape)

    fft = np.fft.fftn
    ifft = np.fft.ifftn
    cross = ifft(
        fft(p_flat, axes=tuple(range(1, 1 + len(core))))
        * np.conj(fft(r_flat, axes=tuple(range(1, 1 + len(core))))),
        axes=tuple(range(1, 1 + len(core))),
    ).real

    shifts: list[np.ndarray] = []
    residuals = []
    for i in range(cross.shape[0]):
        peak = np.unravel_index(np.argmax(cross[i]), img_shape)
        # Wrap to signed shift if not periodic.
        signed = []
        for ax, n in enumerate(img_shape):
            s = peak[ax]
            if not periodic and s > n // 2:
                s = s - n
            signed.append(s)
        shifts.append(np.array(signed))
        # residual RMSE after applying the shift to pred.
        aligned = np.roll(
            p_flat[i],
            shift=tuple(-int(s) for s in signed),
            axis=tuple(range(len(core))),
        )
        residuals.append(float(np.sqrt(np.nanmean((aligned - r_flat[i]) ** 2))))

    shifts_arr = np.stack(shifts)  # (rest, n_dims)
    rest_dims = tuple(rest)
    coords = {
        name: coord
        for name, coord in da_p.coords.items()
        if set(coord.dims).issubset(set(rest_dims))
    }
    out = xr.Dataset(coords=coords)
    for ax, d in enumerate(core):
        arr = (
            shifts_arr[:, ax].reshape(rest_shape)
            if rest_shape
            else int(shifts_arr[0, ax])
        )
        out[f"shift_{d}"] = xr.DataArray(np.asarray(arr), dims=rest_dims)
    res_arr = (
        np.array(residuals).reshape(rest_shape) if rest_shape else float(residuals[0])
    )
    out["residual_rmse"] = xr.DataArray(np.asarray(res_arr), dims=rest_dims)
    return out


# ---------- Centroid displacement ----------------------------------------


def centroid_displacement(
    objects_pred: xr.Dataset,
    objects_ref: xr.Dataset,
    *,
    dims: tuple[str, str] = ("lat", "lon"),
) -> xr.Dataset:
    """Distance between paired object centroids.

    Inputs are labelled-object datasets each carrying a single
    integer-labelled :class:`xr.DataArray` (one variable, ``label``
    field; 0 = background). Pairing is done by label id: label
    ``k > 0`` in ``objects_pred`` is paired with label ``k`` in
    ``objects_ref`` if both exist.

    Args:
        objects_pred: Dataset with a ``"label"`` variable (integer).
        objects_ref: Same shape & dims, ``"label"`` variable.
        dims: Pair of physical dims used to compute centroids
            (default ``("lat", "lon")``).

    Returns:
        Dataset indexed by ``"object"`` with
        ``("displacement_lat", "displacement_lon", "distance",
        "object_id")``. Pairs missing in one side are dropped.
    """
    if len(dims) != 2:
        raise ValueError("centroid_displacement requires two physical dims.")
    if "label" not in objects_pred.data_vars or "label" not in objects_ref.data_vars:
        raise ValueError(
            "Inputs must each carry a 'label' integer DataArray; got "
            f"pred={list(objects_pred.data_vars)}, ref={list(objects_ref.data_vars)}."
        )
    for d in dims:
        if d not in objects_pred["label"].dims or d not in objects_ref["label"].dims:
            raise ValueError(
                f"label DataArrays must carry both dims {dims!r}; got "
                f"pred dims={objects_pred['label'].dims}, "
                f"ref dims={objects_ref['label'].dims}."
            )

    # Force axis order (dims[0], dims[1]) so np.where indexing matches
    # the coord lookups below regardless of how the input was authored.
    lab_p = objects_pred["label"].transpose(*dims).values
    lab_r = objects_ref["label"].transpose(*dims).values
    coord_a = objects_pred.coords[dims[0]].values
    coord_b = objects_pred.coords[dims[1]].values

    def _centroids(lab: np.ndarray) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        ids = np.unique(lab)
        ids = ids[ids > 0]
        for i in ids:
            sel = lab == i
            # Average coord values within the labelled region.
            idx_a, idx_b = np.where(sel)
            out[int(i)] = (float(coord_a[idx_a].mean()), float(coord_b[idx_b].mean()))
        return out

    cp = _centroids(lab_p)
    cr = _centroids(lab_r)
    common = sorted(set(cp) & set(cr))

    if not common:
        return xr.Dataset(
            {
                f"displacement_{dims[0]}": (("object",), np.array([], dtype=float)),
                f"displacement_{dims[1]}": (("object",), np.array([], dtype=float)),
                "distance": (("object",), np.array([], dtype=float)),
                "object_id": (("object",), np.array([], dtype=np.int64)),
            }
        )

    da_a = np.array([cp[i][0] - cr[i][0] for i in common])
    db_a = np.array([cp[i][1] - cr[i][1] for i in common])
    dist = np.sqrt(da_a**2 + db_a**2)
    return xr.Dataset(
        {
            f"displacement_{dims[0]}": (("object",), da_a),
            f"displacement_{dims[1]}": (("object",), db_a),
            "distance": (("object",), dist),
            "object_id": (("object",), np.array(common, dtype=np.int64)),
        }
    )


# ---------- Layer-1 -------------------------------------------------------


class SSIM(Operator):
    """Structural-similarity-index operator (scikit-image)."""

    def __init__(
        self,
        variable: str,
        dims: str | Sequence[str],
        *,
        window: int | None = None,
    ) -> None:
        self.variable = variable
        self.dims = list(_normalize_dims(dims))
        self.window = window

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return ssim(
            ds_pred,
            ds_ref,
            variable=self.variable,
            dims=self.dims,
            window=self.window,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": list(self.dims),
            "window": self.window,
        }


class GradientDifference(Operator):
    """RMS gradient-difference operator."""

    def __init__(self, variable: str, dims: str | Sequence[str]) -> None:
        self.variable = variable
        self.dims = list(_normalize_dims(dims))

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return gradient_difference(
            ds_pred, ds_ref, variable=self.variable, dims=self.dims
        )

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "dims": list(self.dims)}


class PhaseShiftError(Operator):
    """FFT cross-correlation phase-shift operator."""

    def __init__(
        self,
        variable: str,
        dims: str | Sequence[str],
        *,
        periodic: bool = False,
    ) -> None:
        self.variable = variable
        self.dims = list(_normalize_dims(dims))
        self.periodic = periodic

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.Dataset:
        return phase_shift_error(
            ds_pred,
            ds_ref,
            variable=self.variable,
            dims=self.dims,
            periodic=self.periodic,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": list(self.dims),
            "periodic": self.periodic,
        }


class CentroidDisplacement(Operator):
    """Centroid-displacement operator on labelled-object datasets."""

    def __init__(self, dims: tuple[str, str] = ("lat", "lon")) -> None:
        self.dims = tuple(dims)

    def _apply(self, objects_pred: xr.Dataset, objects_ref: xr.Dataset) -> xr.Dataset:
        return centroid_displacement(objects_pred, objects_ref, dims=self.dims)

    def get_config(self) -> dict[str, Any]:
        return {"dims": list(self.dims)}


__all__ = [
    "SSIM",
    "CentroidDisplacement",
    "GradientDifference",
    "PhaseShiftError",
    "centroid_displacement",
    "gradient_difference",
    "phase_shift_error",
    "ssim",
]
