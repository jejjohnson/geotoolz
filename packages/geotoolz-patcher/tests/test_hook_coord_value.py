"""Hook payload extension — coord_value reaches new hooks, legacy 1-arg
hooks still receive only `anchor` without warnings."""

from __future__ import annotations

import warnings

import numpy as np

from geopatcher.time import (
    TemporalCausalBoxcar,
    TemporalFixedLookback,
    TemporalForecast,
    TemporalPatcher,
    TemporalRegularStride,
    TemporalStencilGeometry,
    TemporalStencilSampler,
    TimeStencil,
)


class _LegacyHook:
    """Pre-coord protocol — receives only `anchor`."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def on_patch_start(self, anchor: object) -> None:
        self.calls.append(anchor)


class _CoordHook:
    """New-protocol hook — receives `(anchor, coord_value)`."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def on_patch_start(self, anchor: object, coord_value: object) -> None:
        self.calls.append((anchor, coord_value))


def _stencil_patcher() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalStencilGeometry(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        ),
        sampler=TemporalStencilSampler(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
        ),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalForecast(horizon=1),
    )


def _integer_patcher() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=3),
        sampler=TemporalRegularStride(step=4),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalForecast(horizon=1),
    )


def test_coord_hook_receives_resolved_coord_value_on_stencil_path() -> None:
    coord = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")
    arr = np.arange(coord.shape[0], dtype=np.float32)
    hook = _CoordHook()
    patcher = _stencil_patcher()
    list(patcher.split(arr, hooks=[hook], coord=coord))
    assert hook.calls
    for anchor, value in hook.calls:
        assert value == coord[int(anchor)]


def test_coord_hook_receives_none_on_integer_path() -> None:
    arr = np.arange(24, dtype=np.float32)
    hook = _CoordHook()
    list(_integer_patcher().split(arr, hooks=[hook]))
    assert hook.calls
    for _, value in hook.calls:
        assert value is None


def test_legacy_single_arg_hook_unchanged_on_integer_path() -> None:
    arr = np.arange(24, dtype=np.float32)
    hook = _LegacyHook()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning would fail the test
        list(_integer_patcher().split(arr, hooks=[hook]))
    assert hook.calls  # callbacks did fire


def test_legacy_single_arg_hook_unchanged_on_stencil_path() -> None:
    # The extension is backwards-compatible: the dispatcher trims the
    # trailing coord_value when the hook only declares (anchor,).
    coord = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")
    arr = np.arange(coord.shape[0], dtype=np.float32)
    hook = _LegacyHook()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(_stencil_patcher().split(arr, hooks=[hook], coord=coord))
    assert hook.calls
