"""Operator coupling: which dims an operator's output reads across.

An operator ``T`` *couples* a dim ``d`` when some output depends on two or
more inputs that differ only along ``d``: a filter, a band ratio, a
composite. Re-indexing alone (a band selection or reordering, a crop) moves
values without mixing them, so it is an effect on ``d``, not a coupling.
The set of coupled dims is the operator's coupling class: none is
pointwise, ``band`` is spectral, ``space`` (``y`` and ``x`` together) is
spatial, ``time`` is temporal, and so on.

Two halves live here:

* `Coupling`, the frozen record an operator class declares as its
  ``coupling`` attribute (public in `geoops.carrier`);
* `measure_coupling` / `measure_frames`, which measure it by perturbing a
  sample input, and `assert_coupling`, which holds a declaration to a
  measurement (public in `geoops.testing`).

A measurement is a lower bound: a rank or topology statistic can ignore one
changed pixel, and a toy input can make an operator the identity. So
`assert_coupling` checks that the declaration contains what was measured,
and equality only when asked.

Design: jejjohnson/geostack ``docs/design/operator_coupling.md``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import prod
from typing import Any

import numpy as np
from geoarray import GeoArray

from geoops._src.contract import time_stack_of


__all__ = [
    "CLASS_NAMES",
    "DIMS",
    "EFFECTS",
    "KINDS",
    "Coupling",
    "Measured",
    "assert_coupling",
    "class_name",
    "measure_coupling",
    "measure_frames",
    "reach_of",
]

#: The dims an operator can couple; ``space`` is ``y`` and ``x`` together.
DIMS: tuple[str, ...] = ("band", "level", "space", "time")
#: What an operator does to a dim. A dim with no effect listed is kept.
EFFECTS: tuple[str, ...] = (
    "keep",
    "select",
    "reduce",
    "mix",
    "expand",
    "split",
    "join",
    "regrid",
)
#: What an operator returns, relative to a raster in.
KINDS: tuple[str, ...] = ("transform", "fan-out", "conversion", "source", "sink")

#: Coupled dims -> the class name.
CLASS_NAMES: Mapping[frozenset[str], str] = {
    frozenset(): "pointwise",
    frozenset({"band"}): "spectral",
    frozenset({"space"}): "spatial",
    frozenset({"time"}): "temporal",
    frozenset({"level"}): "vertical",
    frozenset({"band", "space"}): "spatio-spectral",
    frozenset({"space", "time"}): "spatio-temporal",
    frozenset({"band", "time"}): "spectro-temporal",
    frozenset({"level", "space"}): "volumetric",
    frozenset({"band", "space", "time"}): "spatio-spectro-temporal",
    frozenset({"level", "space", "time"}): "volumetric-temporal",
}


def class_name(dims: Iterable[str]) -> str:
    """The class name of a set of coupled dims: ``{"band"}`` is ``"spectral"``.

    Example:
        >>> class_name({"space", "time"})
        'spatio-temporal'
    """
    key = frozenset(dims)
    return CLASS_NAMES.get(key, "+".join(sorted(key)))


def _dims(value: Iterable[str], what: str) -> frozenset[str]:
    if isinstance(value, str):
        raise TypeError(
            f"Coupling {what} takes a set of dims, not the string {value!r}"
        )
    dims = frozenset(value)
    unknown = sorted(dims - set(DIMS))
    if unknown:
        raise ValueError(f"Coupling {what}: unknown dims {unknown}; use {DIMS}")
    return dims


@dataclasses.dataclass(frozen=True, init=False)
class Coupling:
    """Which dims an operator's output reads across, and what it does to each.

    An operator class declares it as ``coupling``, beside ``backends``.
    `geoops.testing.assert_coupling` checks it against a measurement.

    Args:
        couples: The dims along which an output depends on two or more input
            values once any state is fitted (K_apply).
        effects: What the operator does to each dim it changes (`EFFECTS`);
            a dim left out is kept.
        reach: The spatial reach in pixels at the default settings: the
            largest distance between an output pixel and an input pixel it
            depends on. ``None`` for a global or unbounded reach. ``0``
            unless ``space`` is coupled. An operator whose reach depends on
            a constructor argument also defines ``spatial_reach()``.
        fit: The further dims a fitted state is estimated over, beyond
            ``couples`` (``StandardScaler`` applies pointwise but fits its
            mean and spread over ``space``).
        kind: ``"transform"`` (raster in, raster out), ``"fan-out"`` (a list
            of rasters), ``"conversion"`` (a table, geometry, scalar or
            model), ``"source"`` (no raster in) or ``"sink"``.

    Raises:
        ValueError: An unknown dim, effect or kind; a dim both coupled and
            fitted; a reach that does not fit the coupled dims.

    Example:
        >>> ndvi = Coupling(couples={"band"}, effects={"band": "reduce"})
        >>> ndvi.name, ndvi.effect("band"), ndvi.effect("space")
        ('spectral', 'reduce', 'keep')
        >>> Coupling(fit={"space"}).name
        'pointwise'
    """

    couples: frozenset[str]
    effects: tuple[tuple[str, str], ...]
    reach: int | None
    fit: frozenset[str]
    kind: str

    def __init__(
        self,
        *,
        couples: Iterable[str] = (),
        effects: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        reach: int | None = 0,
        fit: Iterable[str] = (),
        kind: str = "transform",
    ) -> None:
        coupled = _dims(couples, "couples")
        fitted = _dims(fit, "fit")
        pairs = dict(effects.items() if isinstance(effects, Mapping) else effects)
        _dims(pairs, "effects")
        bad = sorted(v for v in pairs.values() if v not in EFFECTS)
        if bad:
            raise ValueError(f"Coupling effects: unknown {bad}; use {EFFECTS}")
        pairs = {d: e for d, e in pairs.items() if e != "keep"}
        if kind not in KINDS:
            raise ValueError(f"Coupling kind {kind!r}: use one of {KINDS}")
        if coupled & fitted:
            raise ValueError(
                f"Coupling: {sorted(coupled & fitted)} both coupled and fitted; "
                "fit lists only the further dims a fit reads"
            )
        if reach is not None and (
            isinstance(reach, bool) or not isinstance(reach, int)
        ):
            raise TypeError(f"Coupling reach must be an int or None, got {reach!r}")
        if reach is not None and reach < 0:
            raise ValueError(f"Coupling reach must be >= 0, got {reach}")
        if "space" in coupled and reach == 0:
            raise ValueError(
                "Coupling: a spatial coupling needs a reach >= 1, or None for global"
            )
        if "space" not in coupled and reach != 0:
            raise ValueError("Coupling: a reach other than 0 needs 'space' in couples")
        object.__setattr__(self, "couples", coupled)
        object.__setattr__(self, "effects", tuple(sorted(pairs.items())))
        object.__setattr__(self, "reach", reach)
        object.__setattr__(self, "fit", fitted)
        object.__setattr__(self, "kind", kind)

    @property
    def name(self) -> str:
        """The class name of ``couples``: ``"pointwise"``, ``"spectral"``, …"""
        return class_name(self.couples)

    def effect(self, dim: str) -> str:
        """What the operator does to ``dim``: ``"keep"`` unless listed."""
        return dict(self.effects).get(dim, "keep")


def reach_of(op: Any) -> int | None:
    """``op``'s spatial reach in pixels; ``None`` for a global one.

    ``op.spatial_reach()`` when the operator defines it (a reach that depends
    on a constructor argument), else ``op.coupling.reach``.

    Raises:
        TypeError: ``op`` declares no `Coupling`.

    Example:
        >>> class Box3:
        ...     coupling = Coupling(couples={"space"}, reach=1)
        >>> reach_of(Box3())
        1
    """
    method = getattr(op, "spatial_reach", None)
    if callable(method):
        return method()
    coupling = getattr(op, "coupling", None)
    if not isinstance(coupling, Coupling):
        raise TypeError(f"{type(op).__name__} declares no Coupling")
    return coupling.reach


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

#: The pixel the local probe perturbs, and the far block the non-local probe
#: perturbs; both on a 16 x 16 (or larger) sample.
_CENTRE = (8, 8)
_CORNER = (slice(0, 3), slice(0, 3))
_ATOL = 1e-9


@dataclasses.dataclass(frozen=True)
class Measured:
    """What a perturbation probe saw. A lower bound on the true coupling.

    Attributes:
        couples: The dims an output moved across, with a fresh operator per
            run (a fit on first call fits on the perturbed input).
        frozen_space: Whether a change far from the centre still moved the
            centre output through an operator fitted once on the unperturbed
            input (``None`` when not measured).
        reach: The largest pixel distance at which a one-pixel change moved
            an output; ``None`` when a far change moved the centre.
        regrids: The output is on another grid than the input.
        kind: What came out: ``"transform"``, ``"fan-out"``,
            ``"conversion"`` or ``"sink"``.
        reads_values: Whether any perturbation moved the output.
        effects: The measured effect on each dim that is not kept.
        time: The time probe: ``"per frame"``, ``"pooled"``, ``"reduce"``,
            ``"rejects stack"``, or ``None`` when not run.
        stochastic: Two identical runs differ, so nothing above is reliable.
        notes: Free-form remarks (bands not read, …).
    """

    couples: frozenset[str] = frozenset()
    frozen_space: bool | None = None
    reach: int | None = 0
    regrids: bool = False
    kind: str = "transform"
    reads_values: bool = True
    effects: Mapping[str, str] = dataclasses.field(default_factory=dict)
    time: str | None = None
    stochastic: bool = False
    notes: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The class name of the measured couples."""
        return class_name(self.couples)


def _arrays(out: Any) -> list[np.ndarray] | None:
    if isinstance(out, GeoArray | np.ndarray):
        return [np.asarray(out, dtype=float)]
    is_list = isinstance(out, list | tuple) and bool(out)
    if is_list and all(isinstance(o, GeoArray | np.ndarray) for o in out):
        return [np.asarray(o, dtype=float) for o in out]
    return None


def _kind(out: Any) -> str:
    if out is None:
        return "sink"
    if isinstance(out, GeoArray | np.ndarray):
        return "transform"
    if _arrays(out) is not None:
        return "fan-out"
    return "conversion"


def _equal(a: Any, b: Any) -> bool:
    if hasattr(a, "equals") and not isinstance(a, np.ndarray):
        return bool(a.equals(b))
    if isinstance(a, float | int | np.floating | np.integer):
        return bool(np.isclose(a, b, equal_nan=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, GeoArray | np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        return aa.shape == bb.shape and bool(np.allclose(aa, bb, equal_nan=True))
    if hasattr(a, "data") and hasattr(b, "data"):
        return _equal(np.asarray(a.data), np.asarray(b.data))
    return bool(a == b)


def _changed(a: Any, b: Any) -> np.ndarray | bool:
    """Element-wise "moved" for raster outputs; one bool for anything else."""
    aa, bb = _arrays(a), _arrays(b)
    if aa is None or bb is None:
        try:
            return not _equal(a, b)
        except Exception:
            return True
    if len(aa) != len(bb) or any(
        x.shape != y.shape for x, y in zip(aa, bb, strict=True)
    ):
        return True
    moved = [
        (np.abs(x - y) > _ATOL) | (np.isnan(x) != np.isnan(y))
        for x, y in zip(aa, bb, strict=True)
    ]
    return moved[0] if len(moved) == 1 else np.stack(moved)


def _bump(values: np.ndarray, index: tuple[Any, ...]) -> np.ndarray:
    out = values.copy()
    if out.dtype == bool:
        out[index] = ~out[index]
    elif out.dtype.kind in "iu":
        out[index] = out[index] ^ 0b1111  # flips the low QA bits and the label
    else:
        out[index] = out[index] + 0.5
    return out


def _far_variants(flat: np.ndarray) -> list[np.ndarray]:
    bumped = flat.copy()
    for c in range(flat.shape[0]):
        bumped[c][_CORNER] = _bump(flat[c], _CORNER)[_CORNER]
    if flat.dtype.kind != "f":
        return [bumped]
    dark, bright = flat.copy(), flat.copy()
    dark[(slice(None), *_CORNER)] *= 0.05
    bright[(slice(None), *_CORNER)] += 2.0
    return [bumped, dark, bright]


def _with(x: Any, values: np.ndarray) -> Any:
    return x.replace(data=values) if isinstance(x, GeoArray) else values


def _quiet(fn: Callable[[], Any]) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _same_grid(x: Any, out: Any) -> bool:
    arrays = _arrays(out)
    if arrays is None or len(arrays) != 1:
        return False
    if arrays[0].shape[-2:] != np.shape(x)[-2:]:
        return False
    if isinstance(out, GeoArray) and isinstance(x, GeoArray):
        return bool(np.allclose(np.asarray(out.transform), np.asarray(x.transform)))
    return True


def measure_coupling(
    build: Callable[[], Any],
    sample: Any,
    *others: Any,
    call: Callable[[Any, Any], Any] | None = None,
) -> Measured:
    """Measure which dims ``build()``'s output reads across, on ``sample``.

    Four probes, each on a fresh operator from ``build`` (so a fit on first
    call fits on the perturbed input):

    1. bump one band at the centre pixel, for each band: which outputs move
       gives the band dependence and the spatial reach;
    2. bump, darken and brighten a 3 x 3 block in the far corner: if the
       centre output moves, the operator reads the whole scene;
    3. probe 2 through an operator called once on ``sample`` first, so a
       state fitted on first call is frozen (``frozen_space``);
    4. on a two-frame ``(T, C, H, W)`` stack of a 3-D GeoArray ``sample``,
       bump frame 0: if frame 1's output moves, the operator pools over
       ``time``.

    Args:
        build: Builds the operator, with no arguments (a lambda).
        sample: A raster the operator accepts, at least 16 x 16; a
            ``(C, H, W)`` GeoArray probes ``band`` and ``time`` too.
        *others: Further carriers of a multi-input operator, passed
            unchanged on every call.
        call: ``call(op, x)`` instead of ``op(x, *others)``, for further
            carriers built from ``x``.

    Returns:
        The `Measured` coupling: a lower bound.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> scene = GeoArray(np.ones((2, 16, 16)), (10, 0, 0, 0, -10, 0), 32629,
        ...                  dims="band y x")
        >>> measure_coupling(lambda: (lambda x: x.map(np.sqrt)), scene).name
        'pointwise'
    """
    if np.shape(sample)[-2] <= _CENTRE[0] + 3 or np.shape(sample)[-1] <= _CENTRE[1] + 3:
        raise ValueError("measure_coupling needs a sample of at least 12 x 12 pixels")

    def run(op: Any, x: Any) -> Any:
        return _quiet(lambda: call(op, x) if call is not None else op(x, *others))

    values = np.asarray(sample)
    y0 = run(build(), sample)
    kind = _kind(y0)
    if bool(np.any(_changed(run(build(), sample), y0))):
        return Measured(kind=kind, stochastic=True, notes=("output varies run to run",))
    grid_kept = _same_grid(sample, y0)
    lead = values.shape[:-2]
    n_lead = prod(lead)
    flat = values.reshape(n_lead, *values.shape[-2:])
    dims = tuple(getattr(sample, "dims", ()) or ())
    lead_dim = "level" if "level" in dims else "band"

    rows: dict[int, set[int]] = {}
    moved_per_band: list[bool] = []
    reach = 0
    for c in range(n_lead):
        bumped = _bump(flat, (c, *_CENTRE)).reshape(values.shape)
        moved = _changed(y0, run(build(), _with(sample, bumped)))
        moved_per_band.append(bool(np.any(moved)))
        if isinstance(moved, np.ndarray) and grid_kept and moved.ndim >= 2:
            by_row = moved.reshape(-1, *moved.shape[-2:])
            ys, xs = np.nonzero(by_row.any(axis=0))
            if ys.size:
                dist = np.maximum(abs(ys - _CENTRE[0]), abs(xs - _CENTRE[1]))
                reach = max(reach, int(dist.max()))
            for k in np.nonzero(by_row[:, _CENTRE[0], _CENTRE[1]])[0]:
                rows.setdefault(int(k), set()).add(c)

    global_fresh = global_frozen = False
    if grid_kept:
        for far in _far_variants(flat):
            far_x = _with(sample, far.reshape(values.shape))
            moved = _changed(y0, run(build(), far_x))
            global_fresh |= _centre_moved(moved)
            fitted = build()
            first = run(fitted, sample)
            global_frozen |= _centre_moved(_changed(first, run(fitted, far_x)))

    reads_values = any(moved_per_band) or global_fresh
    couples: set[str] = set()
    effects: dict[str, str] = {}
    notes: list[str] = []
    if kind != "transform":
        return Measured(kind=kind, reads_values=reads_values, reach=0)

    out_lead = prod(np.shape(y0)[:-2])
    if n_lead > 1:
        mixes = any(len(v) >= 2 for v in rows.values())
        moves = out_lead == n_lead and any(k != c for k, v in rows.items() for c in v)
        read = sum(moved_per_band)
        if mixes or (not grid_kept and read >= 2 and out_lead < n_lead):
            couples.add(lead_dim)
        if out_lead == n_lead:
            effect = "mix" if mixes else "select" if moves else "keep"
        elif out_lead < n_lead:
            effect = "reduce" if lead_dim in couples else "select"
        else:
            effect = "expand"
        if effect != "keep":
            effects[lead_dim] = effect
        if read < n_lead:
            notes.append(f"reads {read} of {n_lead} {lead_dim}s")

    measured_reach: int | None = reach
    if grid_kept:
        if reach > 0 or global_fresh:
            couples.add("space")
        if global_fresh:
            measured_reach = None
    else:
        # a new grid: no per-pixel map to read, so no spatial coupling is
        # measured (a lower bound); the effect is what the probe can see
        effects["space"] = "regrid"

    time = _probe_time(build, sample, run)
    if time in ("pooled", "reduce"):
        couples.add("time")
        if time == "reduce":
            effects["time"] = "reduce"
    return Measured(
        couples=frozenset(couples),
        frozen_space=global_frozen if grid_kept else None,
        reach=measured_reach,
        regrids=not grid_kept,
        kind=kind,
        reads_values=reads_values,
        effects=effects,
        time=time,
        notes=tuple(notes),
    )


def _centre_moved(moved: Any) -> bool:
    if isinstance(moved, np.ndarray) and moved.ndim >= 2:
        by_row = moved.reshape(-1, *moved.shape[-2:])
        return bool(by_row[:, _CENTRE[0], _CENTRE[1]].any())
    return bool(moved) if isinstance(moved, bool | np.bool_) else False


def _probe_time(
    build: Callable[[], Any], sample: Any, run: Callable[[Any, Any], Any]
) -> str | None:
    if not isinstance(sample, GeoArray) or sample.ndim != 3 or "level" in sample.dims:
        return None
    stack = time_stack_of(sample)
    try:
        y0 = run(build(), stack)
    except Exception:
        return "rejects stack"
    values = np.asarray(stack).copy()
    values[0] = _bump(values[0], (slice(None), *_CENTRE))
    moved = _changed(y0, run(build(), stack.replace(data=values)))
    if not isinstance(moved, np.ndarray) or moved.ndim < 3:
        return None
    if np.shape(y0)[0] != 2 or np.ndim(y0) < 3:
        return "reduce"
    return "pooled" if bool(moved[1].any()) else "per frame"


def measure_frames(
    build: Callable[[], Any], frames: Sequence[Any], *, as_list: bool = False
) -> Measured:
    """Measure an N-ary reducer: which input frames reach its output.

    Each frame in turn gets its centre pixel bumped. Two or more frames
    reaching the output is a ``time`` coupling; one is a selection.

    Args:
        build: Builds the operator, with no arguments.
        frames: The inputs, one per frame; an item may be a tuple whose
            first element is the raster (``(frame, mask)``).
        as_list: Call ``op(frames)`` instead of ``op(*frames)``.

    Returns:
        The `Measured` coupling along ``time`` (or a band join).

    Example:
        >>> import numpy as np
        >>> frames = [np.full((16, 16), float(t)) for t in range(3)]
        >>> mean = lambda: (lambda *xs: np.mean(np.stack(xs), axis=0))
        >>> measure_frames(mean, frames).name
        'temporal'
    """

    def run(items: Sequence[Any]) -> Any:
        op = build()
        return _quiet(lambda: op(list(items)) if as_list else op(*items))

    y0 = run(frames)
    first = frames[0][0] if isinstance(frames[0], tuple) else frames[0]
    arrays = _arrays(y0)
    in_lead = prod(np.shape(first)[:-2])
    if arrays is not None and len(arrays) == 1:
        out_lead = prod(arrays[0].shape[:-2])
        if out_lead == len(frames) * in_lead > in_lead:
            return Measured(effects={"band": "join"})
    reaches = 0
    for t, item in enumerate(frames):
        frame = item[0] if isinstance(item, tuple) else item
        index = (slice(None), *_CENTRE) if np.ndim(frame) == 3 else _CENTRE
        new = _with(frame, _bump(np.asarray(frame), index))
        bumped = list(frames)
        bumped[t] = (new, *item[1:]) if isinstance(item, tuple) else new
        reaches += bool(np.any(_changed(y0, run(bumped))))
    if arrays is None or len(arrays) != 1:
        effect = "keep"
    else:
        effect = "reduce" if reaches >= 2 else "select"
    return Measured(
        couples=frozenset({"time"}) if reaches >= 2 else frozenset(),
        effects={"time": effect},
        kind=_kind(y0) if arrays is None else "transform",
    )


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def assert_coupling(op: Any, measured: Measured, *, exact: bool = False) -> None:
    """Assert that ``op``'s declared `Coupling` contains what was ``measured``.

    The rules, each a lower bound the declaration must meet:

    * every measured coupled dim is declared, in ``couples`` or ``fit``;
    * a change far away that moved the centre output needs a global
      declaration (``reach=None``), or ``space`` in ``fit`` when only a
      fresh operator moved;
    * a measured reach needs a declared reach at least as large (from
      `reach_of`), unless ``space`` is fitted;
    * a grid change needs ``effects={"space": "regrid"}``;
    * the declared ``kind`` is the measured one (``source`` and ``sink``
      are not measured).

    With ``exact=True`` the declared ``couples | fit`` must also equal the
    measured couples. A stochastic measurement checks nothing.

    Args:
        op: The operator (an instance; a class attribute is read through it).
        measured: From `measure_coupling` or `measure_frames`.
        exact: Require equality, not just containment.

    Raises:
        AssertionError: Naming the operator and every rule it breaks.
    """
    name = type(op).__name__
    declared = getattr(op, "coupling", None)
    if not isinstance(declared, Coupling):
        raise AssertionError(f"{name} declares no `coupling` (geoops.carrier.Coupling)")
    if measured.stochastic:
        return
    problems: list[str] = []
    allowed = declared.couples | declared.fit
    if declared.kind not in ("source", "sink") and measured.kind != declared.kind:
        problems.append(f"kind is {measured.kind!r}, declared {declared.kind!r}")
    if measured.kind == "sink" and declared.kind != "sink":
        problems.append("returns None: declare kind='sink'")
    missing = sorted(measured.couples - allowed)
    if missing:
        problems.append(f"output reads across {missing}, not declared")
    if "space" in measured.couples and "space" in allowed:
        reach = reach_of(op)
        fitted = "space" in declared.fit
        if measured.frozen_space and reach is not None:
            problems.append(
                "a far change moves the centre even with the fit frozen: "
                "declare reach=None"
            )
        if measured.reach is None and reach is not None and not fitted:
            problems.append("a far change moves the centre: declare reach=None")
        if (
            measured.reach is not None
            and reach is not None
            and not fitted
            and measured.reach > reach
        ):
            problems.append(f"measured reach {measured.reach} px > declared {reach} px")
    if measured.regrids and declared.effect("space") != "regrid":
        problems.append("changes the grid: declare effects={'space': 'regrid'}")
    if exact and measured.couples != allowed:
        problems.append(
            f"exact: measured {sorted(measured.couples)}, declared {sorted(allowed)}"
        )
    if problems:
        raise AssertionError(f"{name} coupling: " + "; ".join(problems))
