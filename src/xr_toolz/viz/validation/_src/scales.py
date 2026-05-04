"""V6.2 panels — scale / lead-time / spectral skill (V1 outputs)."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xr_toolz.viz.validation._src.base import _ValidationPanel


class LeadTimeSkillPanel(_ValidationPanel):
    """Plot skill vs forecast lead time.

    Consumes the output of
    :class:`xr_toolz.metrics.SkillByLeadTime` — a
    :class:`xr.DataArray` (or :class:`xr.Dataset`) indexed by
    ``lead_dim`` (default ``"lead_time"``).

    Args:
        lead_dim: Name of the lead-time dimension. Default
            ``"lead_time"``.
        ylabel: Y-axis label. Default ``"Skill"``.
    """

    def __init__(
        self,
        *,
        lead_dim: str = "lead_time",
        ylabel: str = "Skill",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.lead_dim = lead_dim
        self.ylabel = ylabel

    def _default_title(self) -> str:
        return "Skill vs Lead Time"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        skill: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        if isinstance(skill, xr.Dataset):
            for name, da in skill.data_vars.items():
                ax.plot(da[self.lead_dim].values, da.values, marker="o", label=name)
            ax.legend()
        else:
            ax.plot(skill[self.lead_dim].values, skill.values, marker="o")
        ax.set_xlabel(self.lead_dim)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "lead_dim": self.lead_dim,
            "ylabel": self.ylabel,
        }


class ScaleSkillPanel(_ValidationPanel):
    """Bar chart of skill broken down by region.

    Consumes :class:`xr_toolz.metrics.EvaluateByRegion` output — a
    :class:`xr.Dataset` whose data_vars are per-metric scores indexed
    by a ``region`` dimension.

    Args:
        region_dim: Name of the region dimension. Default ``"region"``.
        metric: Name of the variable in the input Dataset to plot.
            ``None`` plots the first data_var.
        ylabel: Y-axis label. Default ``"Score"``.
    """

    def __init__(
        self,
        *,
        region_dim: str = "region",
        metric: str | None = None,
        ylabel: str = "Score",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.region_dim = region_dim
        self.metric = metric
        self.ylabel = ylabel

    def _default_title(self) -> str:
        return "Skill by Region"

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, scores: xr.Dataset | xr.DataArray
    ) -> None:
        if isinstance(scores, xr.Dataset):
            metric_name = self.metric or next(iter(scores.data_vars))
            da = scores[metric_name]
        else:
            da = scores
            metric_name = str(da.name) if da.name is not None else self.ylabel
        regions = [str(r) for r in da[self.region_dim].values]
        values = da.values
        ax = axes
        ax.bar(regions, values, label=metric_name)
        ax.set_xlabel(self.region_dim)
        ax.set_ylabel(self.ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if len(regions) > 6:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "region_dim": self.region_dim,
            "metric": self.metric,
            "ylabel": self.ylabel,
        }


class SpectralSkillPanel(_ValidationPanel):
    """Plot skill vs spatial frequency / wavenumber.

    Consumes a 1-D PSD-score-style :class:`xr.DataArray` indexed by a
    frequency or wavenumber coord (default ``"freq"``). Pairs with
    :class:`xr_toolz.metrics.PSDScore` output collapsed to 1-D.

    Args:
        freq_dim: Frequency / wavenumber dim. Default ``"freq"``.
        log_x, log_y: Whether to use log axes. Default ``log_x=True``,
            ``log_y=False``.
        ylabel: Y-axis label. Default ``"PSD score"``.
    """

    def __init__(
        self,
        *,
        freq_dim: str = "freq",
        log_x: bool = True,
        log_y: bool = False,
        ylabel: str = "PSD score",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.freq_dim = freq_dim
        self.log_x = log_x
        self.log_y = log_y
        self.ylabel = ylabel

    def _default_title(self) -> str:
        return "Skill vs Frequency"

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, scores: xr.DataArray | xr.Dataset
    ) -> None:
        ax = axes
        if isinstance(scores, xr.Dataset):
            for name, da in scores.data_vars.items():
                ax.plot(da[self.freq_dim].values, da.values, label=name)
            ax.legend()
        else:
            freqs = scores[self.freq_dim].values
            ax.plot(np.asarray(freqs), np.asarray(scores.values))
        if self.log_x:
            ax.set_xscale("log")
        if self.log_y:
            ax.set_yscale("log")
        ax.set_xlabel(self.freq_dim)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, which="both", alpha=0.3)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "freq_dim": self.freq_dim,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "ylabel": self.ylabel,
        }


__all__ = ["LeadTimeSkillPanel", "ScaleSkillPanel", "SpectralSkillPanel"]
