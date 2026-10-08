"""The pipekit `Operator` and `GeoTensor` carrier contracts, as assertions.

One implementation of each check, shared by the public
`geotoolz.testing.check_operator` (an operator someone writes on top of the
stack) and the package's own contract suite
(``tests/test_operator_contract.py``, every built-in operator). The rules
are in the repository's ``AGENTS.md`` ("The two contracts").
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any

import numpy as np


def geotensors(value: Any) -> list[Any]:
    """Every GeoTensor in an operator output (a carrier, list or tuple)."""
    from georeader.geotensor import GeoTensor

    if isinstance(value, GeoTensor):
        return [value]
    if isinstance(value, list | tuple):
        return [gt for item in value for gt in geotensors(item)]
    return []


def is_nested_operator(value: Any) -> bool:
    """Whether a config value is a nested-operator payload (or a list of them)."""
    if isinstance(value, list):
        return bool(value) and all(map(is_nested_operator, value))
    return isinstance(value, dict) and set(value) == {"class", "config"}


def same(a: Any, b: Any) -> bool:
    """Config equality: NaN equals NaN, and a tuple never equals a list."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a):
        return bool(np.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        same_kind = type(a) is type(b)
        return same_kind and len(a) == len(b) and all(map(same, a, b))
    return a == b


def assert_keyword_only(cls: type) -> None:
    """``cls.__init__`` takes every parameter by keyword only.

    ``*args`` / ``**kwargs`` pass-throughs (inherited constructors) are
    allowed.
    """
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]
    positional = [
        p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == [], (
        f"{cls.__qualname__}.__init__ takes {positional} positionally; "
        "use __init__(self, *, ...)"
    )


def assert_config_round_trips(op: Any) -> None:
    """``get_config()`` is strict JSON and ``loads(dumps(op))`` rebuilds ``op``.

    pipekit refuses to rebuild a ``forbid_in_yaml`` operator and a
    container whose config nests other operators (those are rebuilt in
    code, or by a YAML / Hydra loader); that refusal is the expected
    outcome for them. A config that is not JSON is a debug repr, allowed
    only on a ``forbid_in_yaml`` operator.
    """
    from pipekit import dumps, loads

    cls = type(op)
    config = op.get_config()
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        if cls.forbid_in_yaml:
            return
        raise AssertionError(
            f"{cls.__name__}.get_config() is not strict JSON ({exc}); hold "
            "only configuration, or set forbid_in_yaml = True for live objects"
        ) from exc
    refused = cls.forbid_in_yaml or any(map(is_nested_operator, config.values()))
    try:
        clone = loads(dumps(op))
    except RuntimeError:
        if refused:
            return
        raise
    assert not refused, f"pipekit rebuilt {cls.__name__}, which it should refuse"
    assert type(clone) is cls, f"loads built {type(clone).__name__}"
    assert same(clone.get_config(), config), (
        f"{cls.__name__} does not round-trip: {config} -> {clone.get_config()}; "
        "store each constructor argument under its own name"
    )


def assert_graph_mode(op: Any, n_inputs: int = 1) -> None:
    """``op`` called on ``Input`` nodes records a graph ``Node``."""
    from pipekit import Input, Node

    node = op(*(Input(f"x{i}") for i in range(n_inputs)))
    assert isinstance(node, Node), (
        f"{type(op).__name__} on Input nodes returned {type(node).__name__}; "
        "implement _apply, never __call__"
    )
    assert node.operator is op
    assert len(node.parents) == n_inputs


def assert_same_carrier(src: Any, out: Any) -> None:
    """A GeoTensor input gives GeoTensors on its CRS; a plain array plain arrays.

    Applies to each carrier of a fan-out (a list of carriers). A carrier on
    the input's pixel grid also keeps its transform.
    """
    from georeader.geotensor import GeoTensor

    from geotoolz._src.geo import grid_matches

    pieces = out if isinstance(out, list | tuple) else [out]
    for piece in pieces:
        if not (isinstance(piece, np.ndarray) and piece.ndim >= 2):
            continue
        if not isinstance(src, GeoTensor):
            assert not isinstance(piece, GeoTensor), (
                "a plain-array input returned a GeoTensor"
            )
            continue
        assert isinstance(piece, GeoTensor), (
            "a GeoTensor input returned a plain array; rewrap with "
            "geotoolz.carrier.wrap_like"
        )
        assert piece.crs == src.crs, f"output CRS {piece.crs} is not {src.crs}"
        if piece.shape[-2:] == src.shape[-2:]:
            assert grid_matches(piece, src), (
                "output on the input's grid has another transform"
            )


def assert_fresh_consistent_attrs(inputs: Any, out: Any) -> None:
    """``out``'s attrs are new dicts whose per-band lists match the band count.

    ``inputs`` is the operator's input (a carrier or a list of carriers).
    Returning an input object itself is allowed (a pass-through).
    """
    from geotoolz._src.bands import PER_BAND_KEYS, band_count

    sources = geotensors(inputs)
    for gt in geotensors(out):
        if any(gt is src for src in sources):
            continue
        assert all(gt.attrs is not src.attrs for src in sources), (
            "output shares an input's attrs dict"
        )
        n_bands = band_count(gt.shape)
        for key in PER_BAND_KEYS:
            value = gt.attrs.get(key)
            if value is None or isinstance(value, str | dict):
                continue
            assert len(value) == n_bands, (
                f"attrs[{key!r}] has {len(value)} entries for {n_bands} band(s)"
            )


def assert_fill_matches_dtype(
    src: Any,
    gt: Any,
    *,
    gap_filler: bool = False,
    nodata: np.ndarray | None = None,
) -> None:
    """``gt.fill_value_default`` suits ``gt``'s dtype and marks ``src``'s nodata.

    * boolean outputs declare ``False``;
    * integer outputs declare an integer their dtype can hold;
    * float outputs declare a fill, and ``NaN`` when ``src`` is an integer
      carrier (an integer fill such as ``0`` collides with promoted data);
    * every nodata pixel of ``src`` is still invalid in an output on the
      same grid (unless the operator fills gaps by design).

    ``nodata`` is the ``(H, W)`` mask of the input pixels that must stay
    invalid; it defaults to every invalid pixel of ``src``.
    """
    from geotoolz._src.valid import _is_nan_scalar, valid_pixels

    fill = gt.fill_value_default
    kind = gt.dtype.kind
    if kind == "b":
        assert isinstance(fill, bool | np.bool_) and not fill, (
            f"bool output declares fill {fill!r}, expected False"
        )
        return
    if kind in "iu":
        assert isinstance(fill, int | np.integer) and not isinstance(fill, bool), (
            f"{gt.dtype} output declares fill {fill!r}, expected an integer"
        )
        assert np.asarray(fill).astype(gt.dtype).item() == fill, (
            f"fill {fill!r} does not fit in {gt.dtype}"
        )
    elif kind == "f":
        assert fill is not None, "float output declares no fill"
        if np.asarray(src).dtype.kind in "biu":
            assert _is_nan_scalar(fill), (
                f"float output of a {np.asarray(src).dtype} input declares fill "
                f"{fill!r}, expected NaN"
            )
    if gap_filler or gt.shape[-2:] != src.shape[-2:]:
        return
    n_fill = int((~valid_pixels(src) if nodata is None else nodata).sum())
    n_invalid = int((~valid_pixels(gt)).sum())
    assert n_invalid >= n_fill, (
        f"input nodata pixels are valid in the output (fill {fill!r}): "
        f"{n_invalid} invalid pixels, expected >= {n_fill}"
    )


def assert_clear_rank_error(name: str, exc: BaseException) -> None:
    """A rejected rank is a ValueError / TypeError / GeoToolzIOError naming ``name``."""
    from geotoolz.io import GeoToolzIOError

    assert isinstance(exc, ValueError | TypeError | GeoToolzIOError), (
        f"library-internal {type(exc).__name__} on a 4-D input: {exc}"
    )
    assert name in str(exc), (
        f"4-D rejection does not name the operator: {type(exc).__name__}: {exc}"
    )


def time_stack_of(scene: Any) -> Any:
    """A 2-frame ``(T, C, H, W)`` stack: ``scene`` and a perturbed copy.

    Float scenes are rescaled per pixel (a fixed seed) so the frames
    differ; integer (label / QA) scenes repeat. A 2-D scene becomes
    ``(T, 1, H, W)``. The stack is on ``scene``'s carrier and grid.
    """
    from geotoolz._src.wrap import wrap_like

    values = np.asarray(scene)
    if values.ndim == 2:
        values = values[None]
    second = values.copy()
    if values.dtype.kind == "f":
        second = values * np.random.default_rng(1).uniform(0.8, 1.2, values.shape)
    return wrap_like(scene, np.stack([values, second]))


def frames(stack: Any) -> list[Any]:
    """The ``(C, H, W)`` frames of a 4-D stack (GeoTensors keep their grid).

    Args:
        stack: A ``(T, C, H, W)`` GeoTensor or ndarray.

    Returns:
        One carrier per time step (``stack.isel({"time": t})`` for a
        GeoTensor, with a copy of its attrs).
    """
    from georeader.geotensor import GeoTensor

    if isinstance(stack, GeoTensor):
        out = []
        for t in range(stack.shape[0]):
            frame = stack.isel({"time": t})
            frame.attrs = dict(stack.attrs or {})
            out.append(frame)
        return out
    return [np.asarray(stack)[t] for t in range(np.shape(stack)[0])]


def assert_time_stack(op: Any, frame: Any) -> None:
    """``op`` on a ``(T, C, H, W)`` stack of ``frame`` matches it per frame.

    The output must equal the per-frame results restacked along time (a
    band-collapsing result keeps a singleton band axis, ``(T, 1, H, W)``),
    or ``op`` must reject the rank with an error naming itself. Each call
    uses a fresh copy of ``op``, so a seeded stochastic operator makes the
    same draws.
    """
    from geotoolz._src.shape import map_frames

    stack = time_stack_of(frame)
    try:
        out = copy.deepcopy(op)(stack)
    except Exception as exc:
        assert_clear_rank_error(type(op).__name__, exc)
        return
    try:
        want = map_frames(lambda f: copy.deepcopy(op)(f), stack, name=type(op).__name__)
    except ValueError:
        return  # per-frame results are not carriers (fan-outs, statistics)
    got, want = np.asarray(out), np.asarray(want)
    assert got.shape == want.shape, (
        f"4-D output shape {got.shape}; per-frame results restack to {want.shape}"
    )
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), equal_nan=True
    )
