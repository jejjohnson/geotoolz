"""V6.4 panel — event verification overlay + contingency summary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.figure as mpl_figure
import matplotlib.patches as mpatches
import xarray as xr

from xr_toolz.viz.validation._src.base import _ValidationPanel


_COLORS = {
    "hit": "#2ca02c",  # matched event in both — green
    "miss": "#1f77b4",  # ref-only — blue
    "false_alarm": "#d62728",  # pred-only — red
}


class EventVerificationPanel(_ValidationPanel):
    """Overlay matched / unmatched events + side panel of scores.

    Consumes ``(objects_pred, objects_ref, matches, scores)`` where:

    - ``objects_pred`` / ``objects_ref`` carry an integer ``label``
      DataArray on ``(lat, lon)`` (0 = background, n = event id).
    - ``matches`` is a mapping with three iterables of (pred_id,
      ref_id) / pred_id / ref_id keyed by ``"hits"``, ``"false_alarms"``,
      ``"misses"``.
    - ``scores`` is a mapping ``{"POD": ..., "FAR": ..., "CSI": ...,
      "IoU": ...}``; rendered as a side text block.

    Cartopy is used when available for a coastline backdrop;
    otherwise a plain matplotlib axes is used.
    """

    _default_axes_layout = (1, 2)

    def __init__(
        self,
        *,
        label_var: str = "label",
        lon: str = "lon",
        lat: str = "lat",
        use_cartopy: bool = True,
        **kw: Any,
    ) -> None:
        kw.setdefault("figsize", (12, 5))
        super().__init__(**kw)
        self.label_var = label_var
        self.lon = lon
        self.lat = lat
        self.use_cartopy = use_cartopy

    def _default_title(self) -> str:
        return "Event verification"

    def _draw_label_outlines(
        self,
        ax: Any,
        labels: xr.DataArray,
        ids: set[int],
        color: str,
        linestyle: str = "-",
    ) -> None:
        if not ids:
            return
        lons = labels[self.lon].values
        lats = labels[self.lat].values
        arr = labels.transpose(self.lat, self.lon).values
        for ev_id in ids:
            mask = (arr == ev_id).astype(float)
            if not mask.any():
                continue
            ax.contour(
                lons,
                lats,
                mask,
                levels=[0.5],
                colors=color,
                linestyles=linestyle,
                linewidths=1.6,
            )

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        objects_pred: xr.Dataset,
        objects_ref: xr.Dataset,
        matches: Mapping[str, Any],
        scores: Mapping[str, float],
    ) -> None:
        ax_map, ax_scores = axes
        if self.use_cartopy:
            try:
                import importlib

                ccrs = importlib.import_module("cartopy.crs")
                proj = ccrs.PlateCarree()
                fig.delaxes(ax_map)
                ax_map = fig.add_subplot(1, 2, 1, projection=proj)
                cfeature = importlib.import_module("cartopy.feature")
                ax_map.add_feature(cfeature.COASTLINE, lw=0.5)
            except (ImportError, ModuleNotFoundError):
                pass

        labels_pred = objects_pred[self.label_var]
        labels_ref = objects_ref[self.label_var]

        hits = matches.get("hits", [])
        hit_pred = {int(p) for p, _ in hits}
        hit_ref = {int(r) for _, r in hits}
        miss_ref = {int(r) for r in matches.get("misses", [])}
        fa_pred = {int(p) for p in matches.get("false_alarms", [])}

        self._draw_label_outlines(ax_map, labels_ref, hit_ref, _COLORS["hit"], "-")
        self._draw_label_outlines(ax_map, labels_pred, hit_pred, _COLORS["hit"], "--")
        self._draw_label_outlines(ax_map, labels_ref, miss_ref, _COLORS["miss"], "-")
        self._draw_label_outlines(
            ax_map, labels_pred, fa_pred, _COLORS["false_alarm"], "--"
        )

        legend_handles = [
            mpatches.Patch(color=_COLORS["hit"], label=f"hits ({len(hits)})"),
            mpatches.Patch(color=_COLORS["miss"], label=f"misses ({len(miss_ref)})"),
            mpatches.Patch(
                color=_COLORS["false_alarm"],
                label=f"false alarms ({len(fa_pred)})",
            ),
        ]
        ax_map.legend(handles=legend_handles, loc="lower left", fontsize=8)
        ax_map.set_xlabel(self.lon)
        ax_map.set_ylabel(self.lat)

        # --- scores side panel ---
        ax_scores.axis("off")
        lines = ["Contingency scores", "-" * 24]
        for name in ("POD", "FAR", "CSI", "IoU"):
            if name in scores:
                lines.append(f"{name:>5}: {float(scores[name]):.3f}")
        for k, v in scores.items():
            if k not in {"POD", "FAR", "CSI", "IoU"}:
                lines.append(f"{k:>5}: {v}")
        ax_scores.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax_scores.transAxes,
            family="monospace",
            va="top",
            fontsize=11,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "label_var": self.label_var,
            "lon": self.lon,
            "lat": self.lat,
            "use_cartopy": self.use_cartopy,
        }


__all__ = ["EventVerificationPanel"]
