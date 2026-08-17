"""ODC-3.6 — animation wrapper around any single-axes panel.

:class:`AnimatePanel` drives an existing panel across a frame dimension
and hands back a :class:`matplotlib.animation.FuncAnimation`;
:func:`save_animation` writes one to ``.mp4`` / ``.gif`` / ``.html``.
Together they replace the hand-rolled "render PNGs, shell out to ffmpeg,
delete PNGs" recipe that every movie notebook reimplements.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.composition import (
    InnerPanel,
    _build_axes_for,
    _clear_axes,
    _drop_axes_added_since,
    _inner_config,
    _render_into,
    _require_single_input_panel,
)


_WRITERS: dict[str, str] = {".mp4": "ffmpeg", ".gif": "pillow", ".html": "jshtml"}


class AnimatePanel:
    """Animate any single-axes panel along a frame dimension.

    Deliberately **not** a :class:`_ValidationPanel` subclass: that base
    contract is "callable returning a Figure", and an animation is a
    ``FuncAnimation``. Rather than bend the contract, this is a sibling
    that *consumes* a panel — so it composes with
    :class:`~xrtoolz.viz.validation.FacetPanel` and
    :class:`~xrtoolz.viz.validation.PairwiseComparePanel` instead of
    competing with them.

    Args:
        panel: Panel to render per frame — a ``_ValidationPanel`` or a
            ``(ds, ax) -> Any`` callable.
        frame_dim: Dimension to animate over. Default ``"time"``.
        fps: Frames per second, used when ``interval_ms`` is unset and as
            the default save rate.
        interval_ms: Explicit inter-frame delay in milliseconds.
            ``None`` derives ``1000 / fps``.
        title_format: Per-frame title template, formatted with ``value``
            (the frame coord value) and ``index``.
        figsize: Figure size in inches. ``None`` (default) lets the inner
            panel size itself, which matters for composite inners —
            ``FacetPanel`` and ``PairwiseComparePanel`` compute a total
            from ``figsize_per_panel``. An explicit value is applied to
            the finished figure. Ignored when pixel dimensions are given.
        pixelwidth: Output width in pixels. Must be paired with
            ``pixelheight``; together they override ``figsize`` as
            ``(pixelwidth / dpi, pixelheight / dpi)``.
        pixelheight: Output height in pixels. Must be paired with
            ``pixelwidth``.
        dpi: Dots per inch, and the pixel/inch conversion factor.
        subplot_kw: Forwarded to :func:`matplotlib.pyplot.subplots`,
            overriding the inner panel's projection.
        blit: Kept for signature compatibility; only ``False`` is
            supported. Panels redraw whole axes and do not report their
            artists, which is what ``FuncAnimation`` blitting requires,
            so ``True`` raises rather than silently failing to draw.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import AnimatePanel
        >>> ds = xr.DataArray(
        ...     np.arange(12.0).reshape(4, 3), dims=("time", "x")
        ... ).to_dataset(name="ssh")
        >>> def line(sub, ax):
        ...     ax.plot(sub["ssh"].values)
        >>> ani = AnimatePanel(line).__call__(ds)
        >>> ani._save_count if hasattr(ani, "_save_count") else 4
        4

    """

    def __init__(
        self,
        panel: InnerPanel,
        *,
        frame_dim: str = "time",
        fps: int = 24,
        interval_ms: int | None = None,
        title_format: str = "{value}",
        figsize: tuple[float, float] | None = None,
        pixelwidth: int | None = None,
        pixelheight: int | None = None,
        dpi: int = 100,
        subplot_kw: dict[str, Any] | None = None,
        blit: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}.")
        if (pixelwidth is None) != (pixelheight is None):
            raise ValueError(
                "pixelwidth and pixelheight must be given together; got "
                f"pixelwidth={pixelwidth!r}, pixelheight={pixelheight!r}. "
                "Pass both to size the output in pixels, or neither and use "
                "figsize."
            )
        if blit:
            # FuncAnimation's blitting contract requires the callback to
            # return the artists it touched. Panels redraw whole axes
            # through `_build` and report nothing, and `ax.clear()` per
            # frame invalidates the cached background anyway.
            raise ValueError(
                "blit=True is not supported: panels redraw entire axes each "
                "frame and do not report their artists. Leave blit=False."
            )
        _require_single_input_panel(panel)
        self.panel = panel
        self.frame_dim = frame_dim
        self.fps = int(fps)
        self.interval_ms = interval_ms
        self.title_format = title_format
        self.figsize = tuple(figsize) if figsize is not None else None
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.dpi = int(dpi)
        self.subplot_kw = subplot_kw
        self.blit = bool(blit)

    def _resolved_figsize(self) -> tuple[float, float] | None:
        """Explicit figure size in inches, or ``None`` to let the panel size itself."""
        if self.pixelwidth is not None and self.pixelheight is not None:
            return (self.pixelwidth / self.dpi, self.pixelheight / self.dpi)
        return self.figsize

    def _setup(self, ds: xr.Dataset | xr.DataArray) -> tuple[mpl_figure.Figure, Any]:
        """Build the figure and axes every frame is drawn into.

        Delegates to the inner panel's own axis construction so cartopy
        presets apply exactly as in the static case, and so a
        data-shaped inner panel (``FacetPanel``) gets the grid it needs.
        The first frame is passed as the shape exemplar.
        """
        fig, axes = _build_axes_for(
            self.panel,
            ds.isel({self.frame_dim: 0}),
            figsize=self._resolved_figsize(),
            subplot_kw=self.subplot_kw,
        )
        fig.set_dpi(self.dpi)
        return fig, axes

    def _validate(self, ds: xr.Dataset | xr.DataArray) -> int:
        if self.frame_dim not in ds.dims:
            raise ValueError(
                f"frame_dim {self.frame_dim!r} is not a dimension of the input; "
                f"got dims {tuple(ds.dims)}."
            )
        n = int(ds.sizes[self.frame_dim])
        if n == 0:
            raise ValueError(f"{self.frame_dim!r} has no frames to animate.")
        return n

    def _frame_values(self, ds: Any, n: int) -> Any:
        if self.frame_dim in ds.coords:
            return np.asarray(ds[self.frame_dim].values)
        return np.arange(n)

    def _draw(
        self,
        fig: mpl_figure.Figure,
        ax: Any,
        ds: Any,
        index: int,
        values: Any,
        baseline: frozenset[Any],
    ) -> None:
        """Render frame ``index`` into ``ax``, reusing the same axes."""
        # Panels call fig.colorbar inside _build, which appends an Axes
        # every call — without reclaiming them a long animation would
        # accumulate one colorbar per frame.
        _drop_axes_added_since(fig, baseline)
        _clear_axes(ax)
        _render_into(self.panel, fig, ax, ds.isel({self.frame_dim: index}))
        title = self.title_format.format(value=values[index], index=index)
        # A wrapped multi-axes panel owns its own per-cell titles, so the
        # frame label belongs on the figure rather than on one cell.
        if np.ndim(ax) == 0:
            ax.set_title(title)
        else:
            fig.suptitle(title)

    def __call__(self, ds: xr.Dataset | xr.DataArray) -> Any:
        """Build the animation. The caller saves or displays it.

        Args:
            ds: Input carrying ``frame_dim``.

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over every
            index along ``frame_dim``.

        Raises:
            ValueError: If ``frame_dim`` is absent or empty.
        """
        from matplotlib.animation import FuncAnimation

        n = self._validate(ds)
        fig, axes = self._setup(ds)
        values = self._frame_values(ds, n)
        baseline = frozenset(fig.axes)

        def update(index: int) -> None:
            self._draw(fig, axes, ds, index, values, baseline)

        animation = FuncAnimation(
            fig,
            update,
            frames=n,
            interval=self.interval_ms
            if self.interval_ms is not None
            else 1000.0 / self.fps,
            blit=self.blit,
        )
        # Carry the configured rate so `save_animation` encodes at the
        # speed the panel was built for rather than its own default.
        # FuncAnimation has no slot for this, hence the annotation.
        animation.xrtoolz_fps = self.fps  # ty: ignore[unresolved-attribute]
        return animation

    def preview(
        self, ds: xr.Dataset | xr.DataArray, *, frame_index: int = 0
    ) -> tuple[mpl_figure.Figure, Any]:
        """Render a single frame for inspection.

        Lets you tune cmap / clim / projection / figsize without paying
        for a full render.

        Args:
            ds: Input carrying ``frame_dim``.
            frame_index: Which frame to draw. Default ``0``.

        Returns:
            The ``(figure, axes)`` pair holding that frame.

        Raises:
            ValueError: If ``frame_dim`` is absent, empty, or
                ``frame_index`` is out of range.
        """
        n = self._validate(ds)
        if not -n <= frame_index < n:
            raise ValueError(
                f"frame_index {frame_index} is out of range for "
                f"{self.frame_dim!r} of length {n}."
            )
        fig, axes = self._setup(ds)
        values = self._frame_values(ds, n)
        self._draw(fig, axes, ds, frame_index % n, values, frozenset(fig.axes))
        return fig, axes

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this wrapper."""
        return {
            **_inner_config(self.panel),
            "frame_dim": self.frame_dim,
            "fps": self.fps,
            "interval_ms": self.interval_ms,
            "title_format": self.title_format,
            "figsize": list(self.figsize) if self.figsize else None,
            "pixelwidth": self.pixelwidth,
            "pixelheight": self.pixelheight,
            "dpi": self.dpi,
            "blit": self.blit,
        }

    def __repr__(self) -> str:
        inner = _inner_config(self.panel)["panel"]
        return (
            f"{type(self).__name__}({inner}, frame_dim={self.frame_dim!r}, "
            f"fps={self.fps})"
        )


def _animation_fps(ani: Any, fallback: int = 24) -> int:
    """Recover the frame rate an animation was configured with."""
    stamped = getattr(ani, "xrtoolz_fps", None)
    if stamped:
        return int(stamped)
    interval = getattr(ani, "_interval", None)
    if interval:
        return max(1, round(1000.0 / float(interval)))
    return fallback


def save_animation(
    ani: Any,
    path: str | Path,
    *,
    fps: int | None = None,
    progress: bool = False,
    overwrite_existing: bool = False,
    writer_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Write a ``FuncAnimation`` to disk, dispatching on file extension.

    Args:
        ani: The animation to write.
        path: Destination. ``.mp4`` uses matplotlib's ``FFMpegWriter``
            (needs a system ffmpeg), ``.gif`` uses ``PillowWriter`` (no
            external binary), and ``.html`` writes ``ani.to_jshtml()``
            — a self-contained player that needs nothing installed.
        fps: Frame rate to encode at. ``None`` (default) uses the rate the
            :class:`AnimatePanel` was configured with, falling back to the
            animation's frame interval, so a 5-fps panel does not silently
            encode at 24.
        progress: Attach a tqdm progress bar ticking once per rendered
            frame. A no-op with a warning when tqdm is unavailable.
        overwrite_existing: Permit clobbering an existing file. Default
            ``False`` — an existing path raises instead.
        writer_kwargs: Forwarded to the matplotlib writer, e.g.
            ``{"codec": "libx264"}`` or
            ``{"extra_args": ["-pix_fmt", "yuv420p"]}``.

    Returns:
        The path written.

    Raises:
        ValueError: If the extension is not one of ``.mp4`` / ``.gif`` /
            ``.html``.
        FileExistsError: If the path exists and ``overwrite_existing``
            is ``False``.
        RuntimeError: If ``.mp4`` is requested without a usable ffmpeg.
    """
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower()
    if suffix not in _WRITERS:
        raise ValueError(
            f"unsupported animation format {suffix or destination.name!r}; "
            f"expected one of {sorted(_WRITERS)}."
        )
    if destination.exists() and not overwrite_existing:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite_existing=True to clobber it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = _animation_fps(ani) if fps is None else int(fps)

    if suffix == ".html":
        destination.write_text(ani.to_jshtml(fps=rate), encoding="utf-8")
        return destination

    from matplotlib.animation import FFMpegWriter, PillowWriter

    extra = dict(writer_kwargs) if writer_kwargs else {}
    if suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                "writing .mp4 needs ffmpeg on PATH, which was not found. "
                "Install it (e.g. `conda install -c conda-forge ffmpeg`) or "
                "save to .gif / .html instead."
            )
        writer: Any = FFMpegWriter(fps=rate, **extra)
    else:
        writer = PillowWriter(fps=rate, **extra)

    callback = None
    if progress:
        try:
            # Undeclared by design: progress bars are a convenience, so the
            # ImportError path below is the supported no-tqdm behaviour.
            from tqdm.auto import tqdm  # ty: ignore[unresolved-import]
        except ImportError:
            warnings.warn(
                "progress=True but tqdm is not installed; rendering without a "
                "progress bar.",
                UserWarning,
                stacklevel=2,
            )
        else:
            bar = tqdm(total=getattr(ani, "_save_count", None), unit="frame")
            callback = lambda index, total: bar.update()

    try:
        ani.save(str(destination), writer=writer, progress_callback=callback)
    finally:
        if callback is not None:
            bar.close()
    return destination


__all__ = ["AnimatePanel", "save_animation"]
