"""Regime-stratified validation bar panels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.figure as mpl_figure
import xarray as xr

from xr_toolz.viz.validation._src.base import _ValidationPanel


class RegionScoreBarPanel(_ValidationPanel):
    """Grouped bar chart for region-stratified metric scores."""

    def __init__(
        self,
        *,
        metrics: Sequence[str] | None = None,
        region_dim: str = "region",
        method_dim: str | None = "method",
        horizontal: bool = False,
        cmap: str = "tab10",
        figsize: tuple[float, float] = (8, 5),
        **kw: Any,
    ) -> None:
        super().__init__(figsize=figsize, **kw)
        self.metrics = tuple(metrics) if metrics is not None else None
        self.region_dim = region_dim
        self.method_dim = method_dim
        self.horizontal = bool(horizontal)
        self.cmap = cmap

    def _default_title(self) -> str:
        return "Scores by Region"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        scores: xr.Dataset | xr.DataArray,
    ) -> None:
        ds = scores.to_dataset() if isinstance(scores, xr.DataArray) else scores
        metrics = list(self.metrics) if self.metrics is not None else list(ds.data_vars)
        subset = ds[metrics]
        ax = axes
        has_method_dim = self.method_dim is not None and self.method_dim in subset.dims
        if has_method_dim:
            df = subset.to_dataframe().unstack(self.method_dim)
        else:
            df = subset.to_dataframe().loc[:, metrics]
        df.index = [str(value) for value in df.index]
        if self.horizontal:
            df.plot.barh(ax=ax, colormap=self.cmap)
            ax.set_xlabel("Score")
            ax.set_ylabel(self.region_dim)
            ax.grid(True, axis="x", alpha=0.3)
        else:
            df.plot.bar(ax=ax, colormap=self.cmap)
            ax.set_xlabel(self.region_dim)
            ax.set_ylabel("Score")
            ax.grid(True, axis="y", alpha=0.3)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "metrics": list(self.metrics) if self.metrics is not None else None,
            "region_dim": self.region_dim,
            "method_dim": self.method_dim,
            "horizontal": self.horizontal,
            "cmap": self.cmap,
        }


__all__ = ["RegionScoreBarPanel"]
