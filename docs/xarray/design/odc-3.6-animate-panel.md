# ODC-3.6 — Generic `AnimatePanel` (time-axis animation wrapper)

**Source survey item:** [ocean-data-challenges-survey.md §3.6](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `movie(ds, name_var, ...)` from `src/mod_plot.py` in `2024c_DC_4DMedSea-ESA`.
**Inspiration:** [`jbusecke/xmovie`](https://github.com/jbusecke/xmovie) (design reuse, not a dependency).

---

## 1. Motivation

The 4DMedSea repo ships a 135-LOC `movie(ds, name_var, ...)` recipe that
renders one frame per time step, saves PNGs to disk, and shells out to
`ffmpeg` to assemble an MP4. The companion `compare_*_uv_png`-style
intercomparison flows do the same side-by-side for two methods.

Both flows are easy enough that they're hand-rolled in every
ocean-data-challenges notebook that wants an animated diagnostic. xr_toolz
has no animation infrastructure today.

This issue adds **one generic panel** —
`AnimatePanel(inner_panel, frame_dim="time")` — that wraps any existing
`_ValidationPanel` (or callable) and produces a `FuncAnimation` over a
chosen frame dim. The intercomparison flow then falls out for free as
`AnimatePanel(PairwiseComparePanel(SpatialMapPanel(...)))` (composing
with ODC-3.4); the upstream's per-method facet animations as
`AnimatePanel(FacetPanel(SpatialMapPanel(...), facet_dim="experiment"))`
(composing with ODC-2.3).

We use `matplotlib.animation.FuncAnimation` directly — no PNG-on-disk
intermediate, no manual subprocess shell-out. Inspiration borrowed from
`xmovie` (preview method, framedim generalization, pixelwidth/pixelheight
sizing, progress + overwrite-guard on save) without taking it as a dep.

## 2. User stories

### 2.1 Animate a spatial SSH map across time (primary)

> *I have a `(time, lat, lon)` SSH Dataset and want an MP4 of one panel
> per time step.*

```python
import xarray as xr
from xr_toolz.viz.validation import (
    SpatialMapPanel, AnimatePanel, save_animation,
)

ds = xr.open_dataset("ssh_daily.nc")    # (time, lat, lon)

panel = AnimatePanel(
    SpatialMapPanel(var="ssh", projection="north_atlantic",
                    cmap="Spectral", vmin=-0.3, vmax=0.3),
    frame_dim="time",
    fps=24,
)
ani = panel(ds)
save_animation(ani, "ssh_2024.mp4", fps=24, progress=True)
```

### 2.2 Preview a single frame before rendering

> *I want to inspect frame 50 to check cmap/clim before committing 30
> minutes to render the full sequence.*

```python
fig, axes = panel.preview(ds, frame_index=50)
fig.savefig("preview.png")
```

### 2.3 Side-by-side method-comparison animation

> *Reproduce the upstream `compare_stat_score_map_uv_png` flow as a
> looping animation.*

```python
from xr_toolz.viz.validation import PairwiseComparePanel

inner = PairwiseComparePanel(
    SpatialMapPanel(var="speed", cmap="viridis", vmin=0, vmax=2.0),
    method_dim="method",
    diff="absolute",
)
ani = AnimatePanel(inner, frame_dim="time", fps=12)(ds)   # ds dims (method, time, lat, lon)
save_animation(ani, "method_comparison.mp4")
```

### 2.4 N-up faceted animation

```python
from xr_toolz.viz.validation import FacetPanel

inner = FacetPanel(
    SpatialMapPanel(var="ssh", projection="gulf_stream", vmin=-0.5, vmax=0.5),
    facet_dim="experiment",
)
ani = AnimatePanel(inner, frame_dim="time")(ds)            # (experiment, time, lat, lon)
save_animation(ani, "experiments.gif")                      # GIF, no ffmpeg needed
```

### 2.5 Inline Jupyter display

```python
from IPython.display import HTML
HTML(panel(ds).to_jshtml())                                 # interactive in-notebook player
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `_ValidationPanel` base + Operator | [`viz/validation/_src/base.py`](../../src/xr_toolz/viz/validation/_src/base.py) | reuse |
| Single-axes panels (SpatialMap, PSD, RegionScoreBar, …) | [`viz/validation/_src/`](../../src/xr_toolz/viz/validation/_src/) | reuse as inner panels |
| `FacetPanel` (N-way over a categorical dim) | proposed in ODC-2.3 | sibling — composable |
| `PairwiseComparePanel` (A-vs-B + diff) | proposed in ODC-3.4 | sibling — composable |
| Time-axis animation wrapper | — | **add** `AnimatePanel` |
| MP4 / GIF / HTML save with progress + overwrite guard | — | **add** `save_animation` |

## 4. Design

### 4.1 Why `matplotlib.animation`, not subprocess + ffmpeg

`matplotlib.animation.FuncAnimation` already provides everything the
upstream `movie` recipe and xmovie hand-roll:

- Frame-by-frame state machine with `update(i)` callback.
- `FFMpegWriter` for MP4 (wraps the ffmpeg binary cleanly — no
  disk-resident PNGs).
- `PillowWriter` for GIF (no ffmpeg required).
- `to_jshtml()` and `to_html5_video()` for inline Jupyter display.

We rely on it directly. The upstream's "render PNG → shell-out → delete
PNG" pattern is replaced by `FuncAnimation.save(...)` internally.
Disk-frame caching is unnecessary.

### 4.2 Generality boundary

Same as `FacetPanel` (ODC-2.3) and `PairwiseComparePanel` (ODC-3.4):

- ✅ Any `_ValidationPanel` subclass — per-frame rendering goes through
  the existing `_build(fig, ax, ds_slice)` hook on the base class.
- ✅ Plain callables `(xr.Dataset, mpl.axes.Axes) -> Any`.
- ❌ Plots that own the whole figure (seaborn `JointGrid`, etc.).
- ❌ Plotly / hvplot / Bokeh — matplotlib only.

Cartopy projections forwarded from the inner panel's `projection`
attribute (same trick as the siblings); explicit `subplot_kw=`
overrides always win.

### 4.3 `AnimatePanel`

```python
# src/xr_toolz/viz/validation/_src/animate.py
class AnimatePanel:
    """Generic time-axis animation wrapper around any single-axes panel.

    **Not** a ``_ValidationPanel`` subclass — the panel base contract is
    "callable returning a Figure", and an animation is fundamentally a
    `FuncAnimation` instead. Reusing the inheritance would break the
    base contract; instead, this is a sibling abstraction that
    *consumes* a `_ValidationPanel` (or callable) and produces an
    animation.

    Builds the figure once via the inner panel's ``_make_fig_axes``;
    on each frame, calls ``inner._build(fig, ax, ds.isel({frame_dim:
    i}))`` — the same render-into-axes hook that ``FacetPanel`` and
    ``PairwiseComparePanel`` use — and updates the per-frame title.
    """

    def __init__(
        self,
        panel: _ValidationPanel | Callable[[xr.Dataset, Any], Any],
        *,
        frame_dim: str = "time",
        fps: int = 24,
        interval_ms: int | None = None,        # default: 1000 / fps
        title_format: str = "{value}",
        figsize: tuple[float, float] = (8, 6),
        pixelwidth: int | None = None,         # xmovie-style; overrides figsize when both set
        pixelheight: int | None = None,
        dpi: int = 100,
        subplot_kw: dict | None = None,
        blit: bool = False,
    ): ...

    def __call__(self, ds: xr.Dataset) -> matplotlib.animation.FuncAnimation:
        """Build and return the FuncAnimation. Caller saves / displays."""
        ...

    def preview(
        self, ds: xr.Dataset, *,
        frame_index: int = 0,
    ) -> tuple[mpl.figure.Figure, Any]:
        """Render a single frame for inspection. Returns (fig, axes).

        Useful for tuning cmap / clim / projection / figsize before
        committing to a full render.
        """
        ...
```

**`__call__` flow**:

1. Validate `frame_dim in ds.dims`.
2. Resolve `figsize` from `pixelwidth`/`pixelheight`/`dpi` if both set
   (`figsize = (pw / dpi, ph / dpi)`).
3. Resolve `subplot_kw` from inner panel's `projection` (when present);
   user `subplot_kw=` wins.
4. Build figure + axes via `inner._make_fig_axes(...)` — pass `figsize`
   and `subplot_kw` through. For `_ValidationPanel` inner, this calls
   the panel's existing axis-creation; for callable inner, plain
   `plt.subplots(figsize=..., subplot_kw=...)`.
5. Define `update(i)`:
   - `ds_slice = ds.isel({frame_dim: i})`.
   - For `_ValidationPanel`: `inner._build(fig, ax, ds_slice)`. Inner
     panel is responsible for `ax.clear()` if needed (existing
     convention).
   - For callable: `inner(ds_slice, ax)`.
   - Set per-frame title via `title_format.format(value=value, index=i)`
     where `value = ds[frame_dim].values[i]`.
6. Return `FuncAnimation(fig, update, frames=ds.sizes[frame_dim],
                          interval=interval_ms or 1000/fps,
                          blit=blit)`.

**`preview(ds, frame_index)` flow**: same as `update(frame_index)` but
without the `FuncAnimation` wrapper. Returns `(fig, axes)` for direct
inspection / `fig.savefig`.

**`get_config`**: recurses into inner panel's `get_config()` when
`_ValidationPanel`; for callables, emits `{"panel": "<callable>"}` flag
indicating non-roundtrippable.

### 4.4 `save_animation` helper

```python
# src/xr_toolz/viz/validation/_src/animate.py
def save_animation(
    ani: matplotlib.animation.FuncAnimation,
    path: str | Path,
    *,
    fps: int = 24,
    progress: bool = False,                # tqdm, optional
    overwrite_existing: bool = False,
    writer_kwargs: dict | None = None,
) -> None:
    """Save a FuncAnimation to disk. Format dispatched on extension.

    - ``.mp4`` → matplotlib.animation.FFMpegWriter (system ffmpeg required)
    - ``.gif`` → matplotlib.animation.PillowWriter
    - ``.html`` → ``ani.to_jshtml()`` written to file

    Parameters
    ----------
    progress
        If True, attach a tqdm progress callback that ticks once per
        rendered frame. No-op if tqdm is not installed (warning logged).
    overwrite_existing
        If False (default) and ``path`` exists, raise FileExistsError
        rather than silently clobbering.
    writer_kwargs
        Forwarded to the matplotlib writer (e.g. ``codec``,
        ``extra_args=["-pix_fmt", "yuv420p"]`` for ffmpeg).
    """
```

**Implementation sketch**:

1. Parse extension; validate against `{".mp4", ".gif", ".html"}`.
2. `if path.exists() and not overwrite_existing`: raise
   `FileExistsError(f"... pass overwrite_existing=True to clobber.")`.
3. For `.mp4`: check ffmpeg via `matplotlib.animation.FFMpegWriter.isAvailable()`;
   raise informative error pointing at `conda install -c conda-forge ffmpeg`
   if missing.
4. Build `progress_callback=lambda i, n: pbar.update()` when `progress=True`
   and tqdm is importable; pass to `ani.save(...)`.
5. Dispatch:
   - `.mp4`: `ani.save(path, writer="ffmpeg", fps=fps, progress_callback=cb, **(writer_kwargs or {}))`.
   - `.gif`: `ani.save(path, writer="pillow", fps=fps, progress_callback=cb, **(writer_kwargs or {}))`.
   - `.html`: write `ani.to_jshtml(fps=fps)` to file.

### 4.5 Composition examples (no new code)

| Upstream concept | Composition |
|---|---|
| `movie(ds, "ssh", method="DUACS")` | `AnimatePanel(SpatialMapPanel(var="ssh"))` |
| `movie_intercomp(ds, "ssh", methods=["DUACS","MIOST"])` | `AnimatePanel(PairwiseComparePanel(SpatialMapPanel(var="ssh"), method_dim="method"))` |
| Per-experiment N-up animation | `AnimatePanel(FacetPanel(SpatialMapPanel(...), facet_dim="experiment"))` |
| Animated PSD-score evolution | `AnimatePanel(PSDIsotropicScorePanel(threshold=0.5), frame_dim="lead_time")` |

Note `frame_dim` is generalized — animate along any dim with monotone
ordering, not just `time`.

## 5. Library leverage

| Need | Library |
|---|---|
| Animation primitive | `matplotlib.animation.FuncAnimation` |
| MP4 writer | `matplotlib.animation.FFMpegWriter` (system ffmpeg required) |
| GIF writer | `matplotlib.animation.PillowWriter` |
| Inline Jupyter HTML | `matplotlib.animation.FuncAnimation.to_jshtml()` |
| Progress bar | `tqdm` (optional; no-op if missing) |
| Cartopy passthrough | inner panel's `projection` |

**No new Python deps.** **System dep**: ffmpeg required for MP4 only;
documented in install docs and surfaced via informative error.

## 6. Public API surface

```python
xr_toolz.viz.validation.AnimatePanel(
    panel,                     # _ValidationPanel | Callable[(ds, ax), Any]
    *,
    frame_dim="time",
    fps=24,
    interval_ms=None,
    title_format="{value}",
    figsize=(8, 6),
    pixelwidth=None, pixelheight=None, dpi=100,
    subplot_kw=None,
    blit=False,
)

xr_toolz.viz.validation.AnimatePanel.preview(ds, *, frame_index=0)
                       -> tuple[Figure, Axes]

xr_toolz.viz.validation.save_animation(
    ani, path, *,
    fps=24,
    progress=False,
    overwrite_existing=False,
    writer_kwargs=None,
)
```

Re-exported from `xr_toolz.viz.validation.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| `AnimatePanel(SpatialMapPanel(var="ssh"))(ds)` returns a `FuncAnimation` | type check |
| Frame count = `ds.sizes[frame_dim]` | exact |
| Per-frame title matches `ds[frame_dim].values[i]` | string match |
| `preview(ds, frame_index=0)` returns `(fig, axes)`; no animation built | type + state |
| `pixelwidth=1920, pixelheight=1080` overrides `figsize` correctly | `figsize == (19.2, 10.8)` at `dpi=100` |
| `save_animation(ani, "out.gif")` works without ffmpeg | passes when ffmpeg unavailable (mock) |
| `save_animation(ani, "out.mp4")` raises informative error if ffmpeg missing | error message mentions install hint (mock) |
| `save_animation(ani, "out.html")` writes HTML inline | file exists, contains `<video>` or jshtml |
| `save_animation(..., overwrite_existing=False)` on existing file | `FileExistsError` |
| `save_animation(..., overwrite_existing=True)` | succeeds, file replaced |
| `save_animation(..., progress=True)` invokes tqdm callback | callback called `n_frames` times |
| Composes with `PairwiseComparePanel` | smoke test on `(method, time, lat, lon)` |
| Composes with `FacetPanel` | smoke test on `(experiment, time, lat, lon)` |
| Cartopy inner panel animates | smoke test with `SpatialMapPanel(projection="gulf_stream")` |
| `get_config` round-trip with `_ValidationPanel` inner | reconstructed panel produces identical `FuncAnimation` (frame count + first-frame match) |
| Dataset without `frame_dim` | informative `ValueError` |
| Callable inner panel (lambda) | works; `get_config` emits non-roundtrippable flag |

Target: ~17 cases.

## 8. Out of scope

- **Dask-parallel frame rendering** (xmovie's `save_frames_parallel`)
  — significant for very long animations; defer to v2 if a user asks.
- **GIF palette generation** for higher-quality GIFs — adds complexity;
  defer.
- **Built-in presets** (rotating-globe-style projection-per-frame
  animations) — projection-per-frame requires writing a custom inner
  panel that mutates `ax.projection` per call; doable but out of scope
  for v1. `SpatialMapPanel(projection=...)` covers static-projection
  animations.
- **Auto-derived `vmin`/`vmax` with warning** — handled at the inner
  panel's constructor; if user omits clim, the inner panel computes
  from frame 0, biasing the colorbar. Documented; no warning machinery
  in `AnimatePanel` itself.
- **3-D / volume animations** — single-axes 2-D inner panels only.
- **Plotly / Bokeh / hvplot animations** — matplotlib only.
- **Real-time / live animations** — file output only.
- **Subprocess shell-out** — `FFMpegWriter` does this cleanly.

## 9. Effort

≈100 LOC implementation + ≈100 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `AnimatePanel` (incl. `pixelwidth`/`pixelheight` resolution) | 70 |
| `preview` method | 10 |
| `save_animation` (progress + overwrite-guard + dispatch) | 25 |
| Tests | ~100 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Naming.** `AnimatePanel` chosen over `MoviePanel` / `TimeAnimation`
   to match sibling pattern (`FacetPanel`, `PairwiseComparePanel`).
   "Movie" is upstream-specific; "Animate" is the matplotlib term.
2. **Return type from `__call__`.** `FuncAnimation` instance, not
   `Figure`. Because the return type diverges from the
   `_ValidationPanel` contract, `AnimatePanel` is a *sibling*
   abstraction (plain class) rather than a subclass — composition over
   inheritance. Save / display happens explicitly via `save_animation`
   or `ani.to_jshtml()`. Document.
3. **Cartopy + animation performance.** Cartopy redraws are expensive.
   `blit=False` default; advanced users can opt into `blit=True` if
   their inner panel supports it (most cartopy panels don't — they
   redraw the whole map). Document.
4. **Inner panel state mutation.** `update(i)` calls
   `inner_panel._build(fig, ax, slice)` which often calls `ax.clear()`
   first. We rely on existing `_build` implementations being idempotent
   — they are (verified by their existing round-trip tests).
5. **`vmin`/`vmax` bias warning.** When user doesn't pass clim to inner
   panel, mpl computes from the *first frame's* data range, which
   becomes the colorbar for the whole animation. Document in
   `AnimatePanel` docstring; recommend always passing `vmin`/`vmax`
   explicitly to the inner panel.
6. **`pixelwidth`/`pixelheight` semantics.** Both must be set to
   override `figsize`; setting one alone is a `ValueError` (ambiguous).
7. **Where it lives.** `viz/validation/_src/animate.py` (new). Sibling
   of `facet.py` and `compare.py`. Could fold into one module; chosen
   separate for clarity (different abstractions).
8. **Operator promotion.** Inheriting from `_ValidationPanel` already
   gives Operator semantics. No extra work.
9. **xmovie-style `save_frames_parallel`** — explicitly deferred.
   `FuncAnimation` renders frames serially; for very long sequences a
   dask-parallel path would help. Track as a v2 enhancement issue if
   demand emerges.
