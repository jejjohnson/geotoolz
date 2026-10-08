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


def is_nan(value: Any) -> bool:
    """Whether ``value`` is a float NaN (``None`` and non-floats are not)."""
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


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
    """``get_config()`` is strict JSON and ``from_state`` rebuilds ``op``.

    A ``forbid_in_yaml`` operator must instead be refused by ``from_state``;
    a container whose config nests other operators is refused too (pipekit
    rebuilds those through a YAML / Hydra loader).
    """
    from pipekit import Operator

    cls = type(op)
    try:
        json.dumps(op.get_config(), allow_nan=False)
    except (TypeError, ValueError) as exc:
        if cls.forbid_in_yaml:
            return  # a debug repr of live objects; never rebuilt from config
        raise AssertionError(
            f"{cls.__name__}.get_config() is not strict JSON ({exc}); hold "
            "only configuration, or set forbid_in_yaml = True for live objects"
        ) from exc
    state = json.loads(json.dumps(op.state))
    nested = any(
        isinstance(v, dict) and set(v) == {"class", "config"}
        for v in op.get_config().values()
    )
    if cls.forbid_in_yaml or nested:
        try:
            Operator.from_state(state)
        except RuntimeError:
            return
        raise AssertionError(
            f"from_state rebuilt {cls.__name__}, which holds live objects; "
            "set forbid_in_yaml = True"
        )
    clone = Operator.from_state(state)
    assert type(clone) is cls, f"from_state built {type(clone).__name__}"
    assert json.dumps(clone.get_config()) == json.dumps(op.get_config()), (
        f"{cls.__name__} does not round-trip: {op.get_config()} -> "
        f"{clone.get_config()}; store each constructor argument under its "
        "own name"
    )


def assert_graph_mode(op: Any, n_inputs: int = 1) -> None:
    """``op`` called on ``Input`` nodes records a graph ``Node``."""
    from pipekit import Input, Node

    node = op(*(Input(f"x{i}") for i in range(n_inputs)))
    assert isinstance(node, Node), (
        f"{type(op).__name__} on Input nodes returned {type(node).__name__}; "
        "implement _apply, never __call__"
    )


def assert_same_carrier(src: Any, out: Any) -> None:
    """A GeoTensor input gives GeoTensors on its CRS; a plain array plain arrays.

    Applies to each carrier of a fan-out (a list of carriers). A carrier on
    the input's pixel grid also keeps its transform.
    """
    from georeader.geotensor import GeoTensor

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
            assert piece.transform == src.transform, (
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
    from geotoolz._src.valid import valid_pixels

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
            assert is_nan(fill), (
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


def assert_time_stack(op: Any, frame: Any) -> None:
    """``op`` on a ``(T, C, H, W)`` stack of ``frame`` matches it per frame.

    The stack holds ``frame`` and a rescaled copy. The output must equal
    the per-frame results restacked along time (a band-collapsing result
    keeps a singleton band axis, ``(T, 1, H, W)``), or ``op`` must reject
    the rank with an error naming itself. Each call uses a fresh copy of
    ``op``, so a seeded stochastic operator makes the same draws.
    """
    from georeader.geotensor import GeoTensor

    values = np.asarray(frame)
    if values.ndim == 2:
        values = values[None]
    second = values * 1.1 if values.dtype.kind == "f" else values.copy()
    if isinstance(frame, GeoTensor):
        stack = GeoTensor(
            np.stack([values, second]),
            transform=frame.transform,
            crs=frame.crs,
            fill_value_default=frame.fill_value_default,
            attrs=dict(frame.attrs),
        )
        frames = [
            GeoTensor(
                v,
                transform=frame.transform,
                crs=frame.crs,
                fill_value_default=frame.fill_value_default,
                attrs=dict(frame.attrs),
            )
            for v in (values, second)
        ]
    else:
        stack, frames = np.stack([values, second]), [values, second]
    try:
        out = copy.deepcopy(op)(stack)
    except Exception as exc:
        assert_clear_rank_error(type(op).__name__, exc)
        return
    per_frame = [np.asarray(copy.deepcopy(op)(f)) for f in frames]
    want = np.stack([p[None] if p.ndim == 2 else p for p in per_frame])
    got = np.asarray(out)
    assert got.shape == want.shape, (
        f"4-D output shape {got.shape}; per-frame results restack to {want.shape}"
    )
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), equal_nan=True
    )
