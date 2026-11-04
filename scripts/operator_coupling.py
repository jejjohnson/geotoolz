"""Measure which dims each geoops operator couples, by perturbing its input.

An operator T couples a dim d when an output at one index depends on an input
at another index along d (or, when d is reduced, on two or more inputs along
it). The set of coupled dims is the operator's coupling class: none is
pointwise, ``band`` is spectral, ``space`` is spatial, ``time`` is temporal,
and so on (`geoops.carrier.Coupling`; geostack
``docs/design/operator_coupling.md``).

For every operator the contract test can build, this script picks the first
toy input the operator accepts and runs `geoops.testing.measure_coupling`
(or `measure_frames` for the N-ary reducers) on it. It reuses the contract
test's tables (constructor kwargs, second carriers, primary inputs, N-ary
inputs), prints a count per measured class and a cross-check against the
contract test's hand lists, and for operators that declare ``coupling``,
whether the declaration holds (`geoops.testing.assert_coupling`).

A measurement is a lower bound: a rank statistic can ignore one pixel, and a
toy input can make an operator the identity (a ``like`` grid equal to the
input grid).

Usage::

    uv run python scripts/operator_coupling.py               # summary
    uv run python scripts/operator_coupling.py --out DIR     # + JSON, table

Run it in the full environment (``make install``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPS_TESTS = ROOT / "packages" / "geotoolz-ops" / "tests"
sys.path.insert(0, str(OPS_TESTS))

import test_operator_contract as tc
from _helpers import toy_geoarray

from geoops.carrier import Coupling
from geoops.testing import Measured, assert_coupling, measure_coupling, measure_frames


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

_LABEL_OPERATOR = "measure._src.operators.LabelConnectedComponents"


def _domain_inputs(key: str) -> list[Any]:
    """Inputs an operator family really reads, tried before the generic ones."""
    scene = tc._scene()
    if key.startswith("mask.") or key == _LABEL_OPERATOR:
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


def probe(cls: type) -> tuple[str, Measured | None, str]:
    """``(status, measurement, why)`` for one operator class."""
    key = tc._key(cls)

    def build() -> Any:
        return tc._build_seeded(cls)

    try:
        op = build()
    except BaseException as exc:  # pytest.skip raises a BaseException
        return "unbuilt", None, str(exc)[:120]
    if key in tc.NARY_INPUTS:
        given = tc.NARY_INPUTS[key]()
        as_list = len(given) == 1 and isinstance(given[0], list)
        frames = given[0] if as_list else given
        try:
            return "ok", measure_frames(build, frames, as_list=as_list), ""
        except Exception as exc:
            return "error", None, f"{type(exc).__name__}: {exc}"
    if not tc._single_carrier_in(op):
        return "no input", None, "no primary input"
    primary = tc.GRAPH_PRIMARY.get(key)
    candidates = [
        *([primary()] if primary is not None else []),
        *_domain_inputs(key),
        *tc._toy_inputs(),
        *tc._attributed_inputs(),
    ]
    why = "no toy input accepted"
    for x in candidates:
        if not tc._is_carrier(x) or np.ndim(x) < 2:
            continue
        try:
            tc._call(build(), x)
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            continue
        try:
            return "ok", measure_coupling(build, x, call=tc._call), ""
        except Exception as exc:
            return "error", None, f"{type(exc).__name__}: {exc}"
    return "no input", None, why[:160]


def report_class(measured: Measured | None) -> str:
    """The row label: the class, or what kept the probe from reading one."""
    if measured is None:
        return "unprobed"
    if measured.stochastic:
        return "stochastic"
    if measured.kind != "transform":
        return measured.kind
    if measured.effects.get("band") == "join":
        return "join"
    if not measured.reads_values:
        return "reads no values"
    name = measured.name
    if not measured.couples and "time" in measured.effects:
        name += f", time {measured.effects['time']}"
    return name


def _declared(cls: type, measured: Measured | None) -> tuple[str, str]:
    coupling = getattr(cls, "coupling", None)
    if not isinstance(coupling, Coupling):
        return "", ""
    if measured is None:
        return coupling.name, "unprobed"
    try:
        assert_coupling(tc._build_seeded(cls), measured)
    except AssertionError as exc:
        return coupling.name, f"FAILS: {str(exc).split(': ', 1)[-1]}"
    return coupling.name, "holds"


def _effects(measured: Measured | None) -> str:
    if measured is None:
        return ""
    parts = [f"{d} {e}" for d, e in sorted(measured.effects.items())]
    if "space" in measured.couples and not measured.regrids:
        parts.append("global" if measured.reach is None else f"reach {measured.reach}")
    if measured.frozen_space is False and measured.reach is None:
        parts.append("fit over space")
    return ", ".join(parts)


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Operator | Measured | Effects | Declared | Check | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r["measured"], r["key"])):
        lines.append(
            f"| `{r['family']}.{r['name']}` | {r['measured']} | {r['effects']} "
            f"| {r['declared']} | {r['check']} | {r['notes'][:80]} |"
        )
    return "\n".join(lines) + "\n"


def _cross_check(rows: list[dict[str, Any]]) -> list[str]:
    """How the measurements compare with the contract test's hand lists."""
    by_key = {r["key"]: r for r in rows}

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
            lambda r: r["time"] == "pooled",
        ),
        report(
            "TIME_INVARIANT",
            sorted(tc.TIME_INVARIANT),
            lambda r: r["measured"] == "reads no values",
        ),
        report(
            "STACK_AS_FEATURES",
            sorted(tc.STACK_AS_FEATURES),
            lambda r: r["time"] == "reduce",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write coupling.json and .md here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    _setup()
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())  # sink operators write their relative paths here
    rows: list[dict[str, Any]] = []
    try:
        for cls in tc._CLASSES:
            if cls.__name__.startswith("_"):
                continue
            status, measured, why = probe(cls)
            declared, check = _declared(cls, measured)
            rows.append(
                {
                    "key": tc._key(cls),
                    "name": cls.__name__,
                    "family": tc._key(cls).split(".")[0],
                    "status": status,
                    "measured": report_class(measured),
                    "effects": _effects(measured),
                    "time": measured.time if measured is not None else None,
                    "declared": declared,
                    "check": check,
                    "notes": "; ".join(measured.notes) if measured else why,
                    "measurement": (
                        dataclasses.asdict(measured) if measured is not None else None
                    ),
                }
            )
    finally:
        os.chdir(cwd)
    print(f"{len(rows)} operators")
    for name, n in Counter(r["measured"] for r in rows).most_common():
        print(f"{n:5d}  {name}")
    for line in _cross_check(rows):
        print(line)
    declared = [r for r in rows if r["declared"]]
    failing = [r for r in declared if r["check"].startswith("FAILS")]
    print(f"declared: {len(declared)}, failing: {len(failing)}")
    for r in failing:
        print(f"  {r['family']}.{r['name']}: {r['check']}")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "coupling.json").write_text(
            json.dumps(rows, indent=1, default=list)
        )
        (args.out / "coupling.md").write_text(_table(rows))
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
