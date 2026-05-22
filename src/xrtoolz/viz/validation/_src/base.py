"""V6.1 — base ``_ValidationPanel`` class.

Validation panels are :class:`Operator` instances that consume one or
more validation-metric outputs (V1–V5) and return a
:class:`matplotlib.figure.Figure`. Subclasses implement
:meth:`_build`; the base class supplies the figure / axes plumbing,
the title / style hooks, and a uniform ``__call__`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
from pipekit import Operator


class _ValidationPanel(Operator):
    """Private base for V6 validation panels.

    Args:
        figsize: Figure size in inches. Default ``(8, 5)``.
        style: Optional matplotlib style name applied via
            :class:`matplotlib.style.context` while the panel renders.
            ``None`` keeps the active rcParams.
        title: Optional panel title; defaults to a class-specific
            string set in :meth:`_default_title`.
        savefig: Optional local output path. When set, the figure is
            saved via :meth:`matplotlib.figure.Figure.savefig` after
            rendering and parent directories are created if missing.
            Cloud / remote paths aren't handled directly — open a
            file-like object yourself and pass it via ``savefig_kwargs``
            if you need that. Default ``None``.
        savefig_kwargs: Forwarded to ``Figure.savefig`` (e.g.
            ``{"dpi": 200, "bbox_inches": "tight"}``). Default ``None``.
        show: When ``True``, call :func:`matplotlib.pyplot.show` after
            rendering — useful for scripts. Notebooks display the
            returned Figure automatically and don't need this. Default
            ``False``.

    Subclasses implement :meth:`_build(fig, axes, *args, **kwargs)`.
    The base ``__call__``:

    - dispatches to graph construction if any positional arg is a
      :class:`~pipekit.Node` (inherited from :class:`Operator`),
    - otherwise creates a Figure + Axes, applies ``style`` if set,
      delegates to :meth:`_build`, applies the title, optionally
      saves and/or shows the figure, and returns it.
    """

    _default_axes_layout: tuple[int, int] = (1, 1)

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (8, 5),
        style: str | None = None,
        title: str | None = None,
        savefig: str | Path | None = None,
        savefig_kwargs: dict[str, Any] | None = None,
        show: bool = False,
    ) -> None:
        self.figsize = tuple(figsize)
        self.style = style
        self.title = title
        self.savefig = savefig
        self.savefig_kwargs = dict(savefig_kwargs) if savefig_kwargs else {}
        self.show = bool(show)

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
        self._maybe_save(fig)
        self._maybe_show(fig)
        return fig

    def _maybe_save(self, fig: mpl_figure.Figure) -> None:
        if self.savefig is None:
            return
        # Create parent dirs for local paths so callers don't have to.
        # Wrap in try/except so non-path-like savefig values (e.g. an
        # open file handle) don't trip Path coercion.
        try:
            parent = Path(self.savefig).expanduser().parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        except (TypeError, OSError):
            pass
        fig.savefig(self.savefig, **self.savefig_kwargs)

    def _maybe_show(self, fig: mpl_figure.Figure) -> None:
        if self.show:
            plt.show()

    def get_config(self) -> dict[str, Any]:
        return {
            "figsize": list(self.figsize),
            "style": self.style,
            "title": self.title,
            "savefig": str(self.savefig) if self.savefig is not None else None,
            "savefig_kwargs": dict(self.savefig_kwargs),
            "show": self.show,
        }


class _NullContext:
    """No-op context manager used when ``style`` is unset."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = ["_ValidationPanel"]
