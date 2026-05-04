"""V6.1 — base ``_ValidationPanel`` class.

Validation panels are :class:`Operator` instances that consume one or
more validation-metric outputs (V1–V5) and return a
:class:`matplotlib.figure.Figure`. Subclasses implement
:meth:`_build`; the base class supplies the figure / axes plumbing,
the title / style hooks, and a uniform ``__call__`` contract.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

from xr_toolz.core import Operator


class _ValidationPanel(Operator):
    """Private base for V6 validation panels.

    Args:
        figsize: Figure size in inches. Default ``(8, 5)``.
        style: Optional matplotlib style name applied via
            :class:`matplotlib.style.context` while the panel renders.
            ``None`` keeps the active rcParams.
        title: Optional panel title; defaults to a class-specific
            string set in :meth:`_default_title`.

    Subclasses implement :meth:`_build(fig, axes, *args, **kwargs)`.
    The base ``__call__``:

    - dispatches to graph construction if any positional arg is a
      :class:`~xr_toolz.core.graph.Node` (inherited from :class:`Operator`),
    - otherwise creates a Figure + Axes, applies ``style`` if set,
      delegates to :meth:`_build`, applies the title, and returns the
      Figure.
    """

    _default_axes_layout: tuple[int, int] = (1, 1)

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (8, 5),
        style: str | None = None,
        title: str | None = None,
    ) -> None:
        self.figsize = tuple(figsize)
        self.style = style
        self.title = title

    def _default_title(self) -> str:
        return self.__class__.__name__

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        nrows, ncols = self._default_axes_layout
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        return fig, axes

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `_build`.")

    def _apply(self, *args: Any, **kwargs: Any) -> mpl_figure.Figure:
        ctx = (
            plt.style.context(self.style) if self.style is not None else _NullContext()
        )
        with ctx:
            fig, axes = self._make_fig_axes()
            self._build(fig, axes, *args, **kwargs)
            title = self.title if self.title is not None else self._default_title()
            # tight_layout before suptitle so suptitle isn't clipped by the
            # axes-only layout pass.
            fig.tight_layout()
            if title:
                fig.suptitle(title)
        return fig

    def get_config(self) -> dict[str, Any]:
        return {
            "figsize": list(self.figsize),
            "style": self.style,
            "title": self.title,
        }


class _NullContext:
    """No-op context manager used when ``style`` is unset."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = ["_ValidationPanel"]
