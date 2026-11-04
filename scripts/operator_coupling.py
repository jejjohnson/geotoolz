"""Measure which dims each geoops operator couples, by perturbing its input.

An operator T couples a dim d when an output at one index depends on an input
at another index along d (or, when d is reduced, on two or more inputs along
it). The set of coupled dims is the operator's coupling class: none is
pointwise, ``band`` is spectral, ``space`` is spatial, ``time`` is temporal,
and so on (geostack ``docs/design/operator_coupling.md``).

The probe runs every operator on a toy input it accepts and perturbs it:

* probe A, local: one pixel at the centre, one band at a time. Which outputs
  move gives the band dependence at that pixel and the spatial reach;
* probe B, non-local: a 3 x 3 block in a far corner, bumped, darkened and
  brightened. If the centre output moves, the operator reads beyond any
  local neighbourhood (a statistic over the scene, a global threshold);
* probe C, apply-only: probe B through an operator already called once on
  the unperturbed input, so a state fitted on first call stays frozen;
* probe T, time: a 2-frame ``(T, C, H, W)`` stack with frame 0 perturbed,
  or, for the N-ary reducers, each input frame in turn.

It reuses the operator contract test's tables (constructor kwargs, second
carriers, primary inputs, N-ary inputs), so it reaches every operator that
test builds. A perturbation that moves nothing is a lower bound: a rank
statistic can ignore one pixel, and a toy input can make an operator the
identity (a ``like`` grid equal to the input grid).

Usage::

    uv run python scripts/operator_coupling.py               # summary
    uv run python scripts/operator_coupling.py --out DIR     # + JSON, table

Run it in the full environment (``make install``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPS_TESTS = ROOT / "packages" / "geotoolz-ops" / "tests"
sys.path.insert(0, str(OPS_TESTS))

import test_operator_contract as tc
from _helpers import toy_geoarray

from geoarray import GeoArray
from geoops._src.contract import time_stack_of


CENTRE = (8, 8)
CORNER = (slice(0, 3), slice(0, 3))
ATOL = 1e-9
#: A spatial reach this far from the centre spans the 16 x 16 toy scene.
GLOBAL_REACH = 7

#: Coupled dims -> class name.
CLASS_NAMES: dict[frozenset[str], str] = {
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
}

#: Probe-only constructor settings, for operators the contract test builds
#: but that cannot run on a toy scene as built.
PROBE_KWARGS: dict[str, dict[str, Any]] = {
    **{
        f"normalize._src.operators.{name}": {"fit_on_call": True}
        for name in ("StandardScaler", "MinMaxScaler", "RobustScaler")
    },
    **{
        f"qa._src.operators.{name}": {"qa_band": 0, "bits": [3]}
        for name in (
            "MaskCirrus",
            "MaskCloudShadow",
            "MaskClouds",
            "MaskSnow",
            "MaskWater",
        )
    },
    "geom._src.operators.CropToBounds": {
        "bounds": (500_030.0, 3_999_870.0, 500_130.0, 3_999_970.0)
    },
}

#: Operators that take one list of carriers, probed like the N-ary reducers.
PROBE_LISTS: tuple[str, ...] = (
    "geom._src.operators.Mosaic",
    "spectral._src.operators.StackBands",
)


# ---------------------------------------------------------------------------
# Comparing outputs
# ---------------------------------------------------------------------------


def _arrays(out: Any) -> list[np.ndarray] | None:
    """The output as numeric arrays, or ``None`` for a table or geometry."""
    if isinstance(out, GeoArray | np.ndarray):
        return [np.asarray(out, dtype=float)]
    items = list(out.values()) if isinstance(out, dict) else out
    if (
        isinstance(items, list | tuple)
        and items
        and all(isinstance(o, GeoArray | np.ndarray) for o in items)
    ):
        return [np.asarray(o, dtype=float) for o in items]
    return None


def _equal(a: Any, b: Any) -> bool:
    if hasattr(a, "equals"):
        return bool(a.equals(b))
    if isinstance(a, float | int | np.floating | np.integer):
        return bool(np.isclose(a, b, equal_nan=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if hasattr(a, "data") and hasattr(b, "data"):
        return _equal(np.asarray(a.data), np.asarray(b.data))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.allclose(a, b, equal_nan=True))
    return bool(a == b)


def _changed(a: Any, b: Any) -> np.ndarray | bool:
    """Element-wise "moved" for array outputs; one bool for anything else."""
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
        (np.abs(x - y) > ATOL) | (np.isnan(x) != np.isnan(y))
        for x, y in zip(aa, bb, strict=True)
    ]
    return moved[0] if len(moved) == 1 else np.stack(moved)


def _centre_moved(moved: Any, grid_kept: bool) -> bool:
    if isinstance(moved, np.ndarray) and grid_kept and moved.ndim >= 2:
        flat = moved.reshape(-1, *moved.shape[-2:])
        return bool(flat[:, CENTRE[0], CENTRE[1]].any())
    return bool(moved) if isinstance(moved, bool | np.bool_) else False


# ---------------------------------------------------------------------------
# Inputs and perturbations
# ---------------------------------------------------------------------------


def _bump(values: np.ndarray, index: tuple) -> np.ndarray:
    """A copy of ``values`` with ``index`` moved by a clearly visible amount."""
    out = values.copy()
    if out.dtype == bool:
        out[index] = ~out[index]
    elif out.dtype.kind in "iu":
        out[index] = out[index] ^ 0b1111  # flips the low QA bits and the label
    else:
        out[index] = out[index] + 0.5
    return out


def _far_variants(flat: np.ndarray) -> list[np.ndarray]:
    """The far corner block bumped, and for float data darkened and brightened."""
    bumped = flat.copy()
    for c in range(flat.shape[0]):
        bumped[c][CORNER] = _bump(flat[c], CORNER)[CORNER]
    if flat.dtype.kind != "f":
        return [bumped]
    dark, bright = flat.copy(), flat.copy()
    dark[(slice(None), *CORNER)] *= 0.05
    bright[(slice(None), *CORNER)] += 2.0
    return [bumped, dark, bright]


def _with(x: Any, values: np.ndarray) -> Any:
    return x.replace(data=values) if isinstance(x, GeoArray) else values


def _domain_inputs(key: str) -> list[Any]:
    """Inputs an operator family really reads, tried before the generic ones."""
    scene = tc._scene()
    if (
        key.startswith("mask.")
        or key == "measure._src.operators.LabelConnectedComponents"
    ):
        return [tc._mask_like(scene)]
    if key.startswith("qa.") and "S2" not in key and "Landsat" not in key:
        qa = np.random.default_rng(0).integers(0, 2**12, (16, 16)).astype(np.uint16)
        return [toy_geoarray(qa, fill_value=None)]
    if key.startswith("restore.") and ("GapFill" in key or "Outlier" in key):
        values = np.asarray(scene).copy()
        if "Outlier" in key:
            values[:, 8, 9] = 50.0  # an outlier next to the centre pixel
        else:
            values[:, 8:10, 9:11] = np.nan  # a hole next to the centre pixel
        return [scene.replace(data=values, fill_value=np.nan)]
    return []


def _setup() -> None:
    for key, kwargs in PROBE_KWARGS.items():
        base = tc.CTOR_KWARGS.get(key, {})
        tc.CTOR_KWARGS[key] = {**(base() if callable(base) else base), **kwargs}
    for key in PROBE_LISTS:
        tc.NARY_INPUTS[key] = lambda: [tc._two_frames()]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _run(op: Any, x: Any) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return tc._call(op, x)


def _fresh(cls: type) -> Any:
    return tc._build_seeded(cls)


def _describe(x: Any, out: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "in_shape": list(np.shape(x)),
        "in_dims": list(getattr(x, "dims", ())) or None,
        "out_type": type(out).__name__,
    }
    arrays = _arrays(out)
    if arrays is None:
        return rec
    rec["out_shape"] = [list(a.shape) for a in arrays]
    same_shape = len(arrays) == 1 and arrays[0].shape[-2:] == np.shape(x)[-2:]
    if isinstance(out, GeoArray) and isinstance(x, GeoArray):
        rec["grid_kept"] = same_shape and bool(
            np.allclose(np.asarray(out.transform), np.asarray(x.transform))
        )
    else:
        rec["grid_kept"] = same_shape
    return rec


def _probe_single(cls: type, x: Any) -> dict[str, Any]:
    """Probes A, B and C on one accepted input ``x``."""
    values = np.asarray(x)
    y0 = _run(_fresh(cls), x)
    rec = _describe(x, y0)
    rec["stochastic"] = bool(np.any(_changed(_run(_fresh(cls), x), y0)))
    grid_kept = bool(rec.get("grid_kept"))
    n_band = prod(values.shape[:-2])
    flat = values.reshape(n_band, *values.shape[-2:])

    deps: list[list[int]] = []  # per input band: output rows moved at the centre
    moved_per_band: list[bool] = []
    reach = 0
    for c in range(n_band):
        bumped = _bump(flat, (c, *CENTRE)).reshape(values.shape)
        moved = _changed(y0, _run(_fresh(cls), _with(x, bumped)))
        moved_per_band.append(bool(np.any(moved)))
        if isinstance(moved, np.ndarray) and grid_kept and moved.ndim >= 2:
            rows_moved = moved.reshape(-1, *moved.shape[-2:])
            ys, xs = np.nonzero(rows_moved.any(axis=0))
            if ys.size:
                dist = np.maximum(abs(ys - CENTRE[0]), abs(xs - CENTRE[1]))
                reach = max(reach, int(dist.max()))
            centre = rows_moved[:, CENTRE[0], CENTRE[1]]
            deps.append([int(k) for k in np.nonzero(centre)[0]])
        else:
            deps.append([])
    rec |= {
        "bands_in": n_band,
        "moved_per_band": moved_per_band,
        "band_deps": deps,
        "reach_px": reach,
        "nonlocal": False,
        "nonlocal_apply": False,
    }
    for far in _far_variants(flat):
        xb = _with(x, far.reshape(values.shape))
        moved = _changed(y0, _run(_fresh(cls), xb))
        rec["nonlocal"] |= _centre_moved(moved, grid_kept)
        op = _fresh(cls)
        first = _run(op, x)
        moved = _changed(first, _run(op, xb))
        rec["nonlocal_apply"] |= _centre_moved(moved, grid_kept)
    return rec


def _probe_time(cls: type, x: Any) -> str | None:
    """Probe T on a 2-frame stack: does perturbing frame 0 move frame 1?"""
    if not isinstance(x, GeoArray) or np.asarray(x).ndim != 3:
        return None
    stack = time_stack_of(x)
    try:
        y0 = _run(_fresh(cls), stack)
    except Exception:
        return "rejects stack"
    values = np.asarray(stack).copy()
    values[0] = _bump(values[0], (slice(None), *CENTRE))
    moved = _changed(y0, _run(_fresh(cls), stack.replace(data=values)))
    if not isinstance(moved, np.ndarray) or moved.ndim < 3:
        return None
    if np.shape(y0)[0] != 2 or np.ndim(y0) < 3:
        return "reduce"
    return "pooled" if bool(moved[1].any()) else "per frame"


def _probe_nary(cls: type) -> dict[str, Any]:
    """Probe T for the N-ary reducers: which input frames reach the output."""
    given = tc.NARY_INPUTS[tc._key(cls)]()
    as_list = len(given) == 1 and isinstance(given[0], list)
    items = given[0] if as_list else given

    def call(xs: list[Any]) -> Any:
        return _fresh(cls)(xs) if as_list else _fresh(cls)(*xs)

    y0 = call(items)
    first = items[0][0] if isinstance(items[0], tuple) else items[0]
    reaches = []
    for t, item in enumerate(items):
        frame = item[0] if isinstance(item, tuple) else item
        index = (slice(None), *CENTRE) if frame.ndim == 3 else CENTRE
        new = frame.replace(data=_bump(np.asarray(frame), index))
        bumped = list(items)
        bumped[t] = (new, *item[1:]) if isinstance(item, tuple) else new
        reaches.append(bool(np.any(_changed(y0, call(bumped)))))
    return {
        "nary": len(items),
        "in_shape": list(np.shape(first)),
        "out_type": type(y0).__name__,
        "out_shape": [list(a.shape) for a in (_arrays(y0) or [])],
        "frames_reaching_output": reaches,
    }


def probe(cls: type) -> dict[str, Any]:
    """Run the probes on ``cls``; ``status`` says whether it could."""
    key = tc._key(cls)
    rec: dict[str, Any] = {"key": key, "name": cls.__name__}
    try:
        op = _fresh(cls)
    except BaseException as exc:  # pytest.skip raises a BaseException
        return rec | {"status": "unbuilt", "why": str(exc)[:120]}
    rec["fittable"] = callable(getattr(op, "fit", None))
    if key in tc.NARY_INPUTS:
        try:
            return rec | {"status": "ok"} | _probe_nary(cls)
        except Exception as exc:
            return rec | {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
    if not tc._single_carrier_in(op):
        return rec | {"status": "no input", "why": "no primary input"}
    primary = tc.GRAPH_PRIMARY.get(key)
    candidates = [
        *([primary()] if primary is not None else []),
        *_domain_inputs(key),
        *tc._toy_inputs(),
        *tc._attributed_inputs(),
    ]
    why = "no toy input accepted"
    for x in candidates:
        if not tc._is_carrier(x):
            continue
        try:
            _run(_fresh(cls), x)
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            continue
        try:
            out = rec | {"status": "ok"} | _probe_single(cls, x)
        except Exception as exc:
            return rec | {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
        out["time_probe"] = _probe_time(cls, x)
        return out
    return rec | {"status": "no input", "why": why[:160]}


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


def _lead(shape: list[int]) -> int:
    return prod(shape[:-2])


def _named(coupled: set[str]) -> str:
    return CLASS_NAMES.get(frozenset(coupled), "+".join(sorted(coupled)))


def classify(r: dict[str, Any]) -> dict[str, Any]:
    """The coupling class, the effect on each dim and notes, from a probe record."""
    out: dict[str, Any] = {
        "key": r["key"],
        "name": r["name"],
        "family": r["key"].split(".")[0],
        "couples": [],
        "effects": {},
        "notes": [],
    }
    if r["status"] != "ok":
        return out | {"cls": "unprobed", "notes": [r.get("why", r["status"])]}
    out_type = r.get("out_type")
    out_shapes = r.get("out_shape") or []
    if out_type not in ("GeoArray", "ndarray", "list", "tuple", "dict"):
        return out | {"cls": "conversion", "notes": [f"to {out_type}"]}
    if out_type in ("list", "tuple") and not r.get("nary"):
        if out_shapes and all(s[-2:] != r["in_shape"][-2:] for s in out_shapes):
            kind = "space split"
        elif len(out_shapes) == r.get("bands_in"):
            kind = "band split"
        else:
            kind = "split"
        return out | {"cls": "fan-out", "notes": [kind]}
    if r.get("nary"):
        return out | _classify_nary(r)
    if (
        len(out_shapes) == 1
        and sorted(out_shapes[0]) == sorted(r["in_shape"])
        and out_shapes[0] != r["in_shape"]
    ):
        return out | {"cls": "rearrange", "notes": ["dims reordered"]}
    if not any(r["moved_per_band"]) and not r["nonlocal"]:
        return out | {"cls": "reads no values", "notes": ["no output moved"]}

    coupled: set[str] = set()
    effects: dict[str, str] = {}
    notes: list[str] = []
    lead_dim = "level" if "level" in (r.get("in_dims") or []) else "band"
    c_in = r["bands_in"]
    if c_in > 1 and out_shapes:
        c_out = _lead(out_shapes[0]) if len(out_shapes) == 1 else None
        rows: dict[int, set[int]] = {}
        for c, moved in enumerate(r["band_deps"]):
            for k in moved:
                rows.setdefault(k, set()).add(c)
        read = sum(r["moved_per_band"])
        mixes = any(len(v) >= 2 for v in rows.values())
        moves = c_out == c_in and any(k != c for k, v in rows.items() for c in v)
        fallback = not rows and read >= 2 and c_out is not None and c_out < c_in
        if mixes or moves or fallback:
            coupled.add(lead_dim)
        if c_out is None:
            effects[lead_dim] = "split"
        elif c_out == c_in:
            effects[lead_dim] = "mix" if lead_dim in coupled else "keep"
        elif c_out < c_in:
            effects[lead_dim] = "reduce" if lead_dim in coupled else "select"
        else:
            effects[lead_dim] = "expand"
        if read < c_in:
            notes.append(f"reads {read} of {c_in} {lead_dim}s")

    if r.get("grid_kept"):
        if r["reach_px"] > 0 or r["nonlocal"]:
            coupled.add("space")
        if r["nonlocal"]:
            effects["space"] = "global" if r["nonlocal_apply"] else "fit over space"
        elif r["reach_px"] >= GLOBAL_REACH:
            effects["space"] = "global (by reach)"
        elif r["reach_px"] > 0:
            effects["space"] = f"local r<={r['reach_px']}"
        else:
            effects["space"] = "keep"
    elif out_shapes:
        coupled.add("space")
        same = out_shapes[0][-2:] == r["in_shape"][-2:]
        effects["space"] = "regrid (same shape)" if same else "regrid"

    time = r.get("time_probe")
    if time in ("pooled", "reduce"):
        coupled.add("time")
    if time:
        effects["time"] = time
    if r.get("stochastic"):
        notes.append("stochastic")
    if r.get("fittable"):
        notes.append("fit/transform")
    return out | {
        "cls": _named(coupled),
        "couples": sorted(coupled),
        "effects": effects,
        "notes": notes,
    }


def _classify_nary(r: dict[str, Any]) -> dict[str, Any]:
    reach = r["frames_reaching_output"]
    out_shapes = r["out_shape"]
    in_lead = _lead(r["in_shape"])
    if len(out_shapes) == 1 and _lead(out_shapes[0]) == r["nary"] * in_lead > in_lead:
        return {"cls": "join", "effects": {"band": "join"}, "notes": ["joins bands"]}
    if len(out_shapes) != 1:
        effect = "keep"
    else:
        effect = "reduce" if sum(reach) >= 2 else "select"
    coupled = {"time"} if sum(reach) >= 2 else set()
    name = _named(coupled) if coupled else f"pointwise, time {effect}"
    return {"cls": name, "couples": sorted(coupled), "effects": {"time": effect}}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _table(classes: list[dict[str, Any]]) -> str:
    lines = [
        "| Operator | Class | band / level | space | time | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in sorted(classes, key=lambda c: (c["cls"], c["family"], c["name"])):
        e = c["effects"]
        lines.append(
            f"| `{c['family']}.{c['name']}` | {c['cls']} "
            f"| {e.get('band', e.get('level', ''))} | {e.get('space', '')} "
            f"| {e.get('time', '')} | {'; '.join(c['notes'])[:80]} |"
        )
    return "\n".join(lines) + "\n"


def _cross_check(classes: list[dict[str, Any]]) -> list[str]:
    """How the measured classes compare with the contract test's hand lists."""
    by_key = {c["key"]: c for c in classes}

    def report(name: str, keys: list[str], measured: Any) -> str:
        hit = [k for k in keys if k in by_key and measured(by_key[k])]
        missed = sorted(k.rsplit(".", 1)[-1] for k in keys if k not in hit)
        return f"{name}: {len(hit)}/{len(keys)} measured" + (
            f" (not: {', '.join(missed)})" if missed else ""
        )

    return [
        report(
            "POOLED_STATISTICS",
            list(tc.POOLED_STATISTICS),
            lambda c: c["effects"].get("time") == "pooled",
        ),
        report(
            "TIME_INVARIANT",
            sorted(tc.TIME_INVARIANT),
            lambda c: c["cls"] == "reads no values",
        ),
        report(
            "STACK_AS_FEATURES",
            sorted(tc.STACK_AS_FEATURES),
            lambda c: c["effects"].get("time") == "reduce",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write coupling.json and .md here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    _setup()
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())  # sink operators write their relative paths here
    try:
        records = [probe(c) for c in tc._CLASSES if not c.__name__.startswith("_")]
    finally:
        os.chdir(cwd)
    classes = [classify(r) for r in records]
    counts = Counter(c["cls"] for c in classes)
    print(f"{len(classes)} operators")
    for name, n in counts.most_common():
        print(f"{n:5d}  {name}")
    for line in _cross_check(classes):
        print(line)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = [r | {"class": c} for r, c in zip(records, classes, strict=True)]
        (args.out / "coupling.json").write_text(json.dumps(payload, indent=1))
        (args.out / "coupling.md").write_text(_table(classes))


if __name__ == "__main__":
    main()
