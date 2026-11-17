"""Leaderboard helper — sorted ``(method, metric)`` DataFrame builder.

Companion to the panel + metric Operators: takes a Dataset of scores
indexed by a ``method`` (and optional region) dim and returns a
sorted ``pandas.DataFrame`` ready for Markdown / LaTeX rendering.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import xarray as xr


def rank_methods(
    scores: xr.Dataset,
    by: str,
    ascending: bool = True,
    include: Sequence[str] | None = None,
    method_dim: str = "method",
    region_dim: str | None = None,
) -> pd.DataFrame:
    """Build a sorted leaderboard DataFrame from a per-method score Dataset.

    Args:
        scores: :class:`xr.Dataset` whose ``data_vars`` are scalar (or
            per-region) metrics, with one dim named ``method_dim``.
        by: Primary sort-key metric (must be a data_var of ``scores``).
        ascending: If ``True``, lower is better. Default ``True``.
        include: Optional subset of metrics to keep in the output (in
            this order). ``None`` keeps all data_vars.
        method_dim: Name of the method dim. Default ``"method"``.
        region_dim: When set, the leaderboard is built per region with
            a ``MultiIndex`` of ``(region, method)``.

    Returns:
        :class:`pandas.DataFrame` indexed by method (or ``(region,
        method)`` when ``region_dim`` is given), sorted by ``by`` then
        by remaining columns as tie-breakers. Render with
        ``df.to_markdown()`` for paper-ready tables.
    """
    if method_dim not in scores.dims:
        raise ValueError(
            f"scores must have a {method_dim!r} dim; got {tuple(scores.dims)}."
        )
    if by not in scores.data_vars:
        raise ValueError(f"sort key {by!r} not in data_vars {list(scores.data_vars)}.")

    keep = list(include) if include is not None else list(scores.data_vars)
    missing = [m for m in keep if m not in scores.data_vars]
    if missing:
        raise ValueError(f"include contains unknown metrics: {missing}.")
    if by not in keep:
        keep = [by, *keep]

    df = scores[keep].to_dataframe()

    # Choose the index columns. If region_dim is given, use both.
    if region_dim is not None:
        if region_dim not in scores.dims:
            raise ValueError(
                f"region_dim {region_dim!r} not in dims {tuple(scores.dims)}."
            )
        df = df.reorder_levels([region_dim, method_dim]).sort_index()
        # Sort within each region by `by`, ties broken by remaining cols.
        sort_cols = [by, *[c for c in keep if c != by]]
        df = df.groupby(level=region_dim, group_keys=False).apply(
            lambda g: g.sort_values(sort_cols, ascending=ascending, kind="stable")
        )
    else:
        if df.index.nlevels > 1:
            extra = [n for n in df.index.names if n != method_dim]
            # Only squeeze *singleton* extra levels — silently dropping
            # multi-valued levels (e.g. forgetting region_dim on a
            # ``(region, method)`` dataset) would mix rows from
            # different groups into one global leaderboard and produce
            # duplicated method entries with misleading ranks.
            non_singleton = [n for n in extra if df.index.unique(level=n).size > 1]
            if non_singleton:
                raise ValueError(
                    f"scores has multi-valued extra index level(s) {non_singleton!r} "
                    f"besides {method_dim!r}; pass region_dim=... (or pre-reduce) so "
                    "the leaderboard isn't built across mixed groups."
                )
            df = df.reset_index(extra, drop=True)
        sort_cols = [by, *[c for c in keep if c != by]]
        df = df.sort_values(sort_cols, ascending=ascending, kind="stable")

    return df


__all__ = ["rank_methods"]
