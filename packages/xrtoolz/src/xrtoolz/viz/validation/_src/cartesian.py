"""Projection-free map panel for Cartesian model output.

:class:`CartesianMapPanel` renders a 2-D snapshot on ``x``/``y`` axes in
metres (shown in km by default) — the idealised f/β-plane output of ocean
models such as somax, which has no lon/lat and needs no cartopy. It
replaces the hand-rolled ``pcolormesh(x/1e3, y/1e3, field, cmap="RdBu_r",
vmin=-q99, vmax=q99)`` pattern, and composes with :class:`FacetPanel`,
:class:`PairwiseComparePanel` and :class:`AnimatePanel` like
:class:`SpatialMapPanel` does.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz._src.cmaps import cmap_for
from xrtoolz.viz._src.norm import shared_norm
from xrtoolz.viz.validation._src.base import _NullContext, _ValidationPanel


NormSpec = Literal["symmetric_q99", "q99"] | tuple[float, float] | None

_UNIT_SCALE: dict[str, float] = {"m": 1.0, "km": 1e-3}
_Q99 = (0.01, 0.99)


class CartesianMapPanel(_ValidationPanel):
    """Single 2-D ``(y, x)`` snapshot on Cartesian axes, without cartopy.

    Args:
        variable: Data-variable name when input is a Dataset. ``None``
            (default) auto-picks the first ``data_var``. Also the lookup
            key for :func:`xrtoolz.viz.cmap_for` when ``cmap`` is unset.
        dims: ``(x_dim, y_dim)`` coordinate names, in metres. Default
            ``("x", "y")``.
        units: Axis units to display, ``"km"`` (default) or ``"m"``.
        time_index: When the input has a ``time`` dim, ``isel(time=…)``
            this index before plotting. Default ``0``; ``None`` skips the
            selection (e.g. for pre-reduced or animated inputs).
        cmap: Matplotlib colormap. ``None`` (default) auto-resolves from
            the variable registry, falling back to ``"viridis"``.
        norm: Colour limits. ``"symmetric_q99"`` (default) centres the
            scale on zero at the 1/99 % quantiles — the convention for
            signed fields such as vorticity or SSH anomaly. ``"q99"``
            clips at the 1/99 % quantiles without centring. A
            ``(vmin, vmax)`` tuple is used as-is; ``None`` lets matplotlib
            span the full range.
        mask: Name of a mask variable or coordinate (non-zero / ``True``
            = valid). Cells where it is zero are blanked. Ignored when the
            input carries no such name. Default ``"mask"``.
        contour: Name of a Dataset variable to overlay as black contour
            lines (e.g. ``"psi"`` over vorticity). Default ``None``.
        contour_levels: Number of contour levels. Default ``10``.
        cbar_label: Colorbar label. ``None`` (default) builds one from the
            variable's ``long_name`` / ``units`` attrs; ``""`` suppresses it.

    Returns:
        A :class:`matplotlib.figure.Figure` with one equal-aspect map.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import CartesianMapPanel
        >>> x = np.linspace(0.0, 1e6, 32)
        >>> zeta = np.sin(x[None, :] / 2e5) * np.cos(x[:, None] / 2e5)
        >>> ds = xr.Dataset({"zeta": (("y", "x"), zeta)}, coords={"x": x, "y": x})
        >>> fig = CartesianMapPanel(variable="zeta")(ds)
        >>> fig.axes[0].get_xlabel()
        'x [km]'

    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        variable: str | None = None,
        dims: tuple[str, str] = ("x", "y"),
        units: Literal["km", "m"] = "km",
        time_index: int | None = 0,
        cmap: str | None = None,
        norm: NormSpec = "symmetric_q99",
        mask: str | None = "mask",
        contour: str | None = None,
        contour_levels: int = 10,
        cbar_label: str | None = None,
        **kw: Any,
    ) -> None:
        if units not in _UNIT_SCALE:
            raise ValueError(f"units={units!r} must be one of {sorted(_UNIT_SCALE)}.")
        if isinstance(norm, list | tuple):
            # Lists arrive from a get_config() round-trip.
            norm = (float(norm[0]), float(norm[1]))
        elif isinstance(norm, str) and norm not in ("symmetric_q99", "q99"):
            raise ValueError(
                f"norm={norm!r} must be 'symmetric_q99', 'q99', a (vmin, vmax) "
                "tuple or None."
            )
        super().__init__(**kw)
        self.variable = variable
        self.dims = tuple(dims)
        self.units = units
        self.time_index = time_index
        self.cmap = cmap
        self.norm = norm
        self.mask = mask
        self.contour = contour
        self.contour_levels = int(contour_levels)
        self.cbar_label = cbar_label

    def _default_title(self) -> str:
        return self.variable or "Snapshot"

    def _select(self, obj: xr.DataArray | xr.Dataset, name: str | None) -> xr.DataArray:
        if isinstance(obj, xr.Dataset):
            da = obj[name or next(iter(obj.data_vars))]
        else:
            da = obj
        if self.time_index is not None and "time" in da.dims:
            da = da.isel(time=self.time_index)
        return da

    def _apply_mask(
        self, obj: xr.DataArray | xr.Dataset, da: xr.DataArray
    ) -> xr.DataArray:
        if self.mask is None:
            return da
        has_mask = self.mask in obj.coords or (
            isinstance(obj, xr.Dataset) and self.mask in obj.data_vars
        )
        if not has_mask:
            return da
        valid = obj[self.mask]
        if self.time_index is not None and "time" in valid.dims:
            valid = valid.isel(time=self.time_index)
        return da.where(valid.astype(bool))

    def _limits(self, da: xr.DataArray) -> tuple[float | None, float | None]:
        if self.norm is None:
            return None, None
        if isinstance(self.norm, tuple):
            return float(self.norm[0]), float(self.norm[1])
        lo, hi = shared_norm(da, q=_Q99, symmetric=self.norm == "symmetric_q99")
        if np.isnan(lo):
            return None, None
        return lo, hi

    def _label(self, da: xr.DataArray) -> str:
        if self.cbar_label is not None:
            return self.cbar_label
        name = da.attrs.get("long_name", da.name or "")
        units = da.attrs.get("units")
        return f"{name} [{units}]" if units else str(name)

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        snapshot: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        x_dim, y_dim = self.dims
        da = self._apply_mask(snapshot, self._select(snapshot, self.variable))
        if da.ndim != 2:
            raise ValueError(
                f"CartesianMapPanel needs a 2-D ({y_dim}, {x_dim}) field after "
                f"time selection; got dims {tuple(da.dims)}."
            )
        da = da.transpose(y_dim, x_dim)
        scale = _UNIT_SCALE[self.units]
        x = np.asarray(da[x_dim].values) * scale
        y = np.asarray(da[y_dim].values) * scale
        vmin, vmax = self._limits(da)
        cmap = self.cmap if self.cmap is not None else cmap_for(self.variable)
        im = ax.pcolormesh(
            x, y, np.asarray(da.values), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        if self.contour is not None:
            line = self._select(snapshot, self.contour).transpose(y_dim, x_dim)
            ax.contour(
                x,
                y,
                np.asarray(line.values),
                levels=self.contour_levels,
                colors="k",
                linewidths=0.6,
            )
        ax.set_xlabel(f"{x_dim} [{self.units}]")
        ax.set_ylabel(f"{y_dim} [{self.units}]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label=self._label(da))

    def _apply(self, *args: Any, **kwargs: Any) -> mpl_figure.Figure:
        # Title on the axes rather than as a suptitle, as SpatialMapPanel does.
        import matplotlib.pyplot as plt

        ctx = (
            plt.style.context(self.style) if self.style is not None else _NullContext()
        )
        with ctx:
            fig, axes = self._make_fig_axes()
            self._build(fig, axes, *args, **kwargs)
            title = self.title if self.title is not None else self._default_title()
            if title:
                axes.set_title(title)
            fig.tight_layout()
        self._maybe_save(fig)
        self._maybe_show(fig)
        return fig

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "variable": self.variable,
            "dims": list(self.dims),
            "units": self.units,
            "time_index": self.time_index,
            "cmap": self.cmap,
            "norm": list(self.norm) if isinstance(self.norm, tuple) else self.norm,
            "mask": self.mask,
            "contour": self.contour,
            "contour_levels": self.contour_levels,
            "cbar_label": self.cbar_label,
        }


__all__ = ["CartesianMapPanel"]
