"""Measure which dims each geoops operator couples, by perturbing its input.

An operator T couples a dim d when an output at one index depends on an input
at another index along d (or, when d is reduced, on two or more inputs along
it). The set of coupled dims is the operator's coupling class: none is
pointwise, ``band`` is spectral, ``space`` is spatial, ``time`` is temporal,
and so on (`geoops.carrier.Coupling`; geostack
``docs/design/operator_coupling.md``).

For every operator the contract test can build, this script picks the first
toy input the operator accepts and runs `geoops.testing.measure_coupling`
(or `measure_frames` for the N-ary reducers) on it, through the same input
selection as ``tests/test_operator_coupling.py``
(``packages/geotoolz-ops/tests/_coupling_inputs.py``). It prints a count
per measured class, a cross-check against the contract test's hand lists,
and, for operators that declare ``coupling``, whether the declaration holds
(`geoops.testing.assert_coupling`); it exits non-zero when one does not.

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


ROOT = Path(__file__).resolve().parents[1]
OPS_TESTS = ROOT / "packages" / "geotoolz-ops" / "tests"
sys.path.insert(0, str(OPS_TESTS))

import _coupling_inputs as ci
import test_operator_contract as tc

from geoops.carrier import Coupling
from geoops.testing import Measured, assert_coupling


def probe(cls: type) -> tuple[str, Measured | None, str]:
    """``(status, measurement, why)`` for one operator class."""
    measured, why = ci.probe(cls)
    return ("ok" if measured is not None else "unprobed"), measured, why


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
        assert_coupling(ci.build_for_probe(cls), measured)
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
    """The contract test's time-stack sets (derived from the declarations)
    against what the probe saw: how many members it confirms, and any
    operator it saw behave that way outside the set."""
    by_key = {r["key"]: r for r in rows}

    def report(name: str, keys: frozenset[str], seen: Any, *, closed: bool) -> str:
        confirmed = sum(1 for k in keys if k in by_key and seen(by_key[k]))
        # a "reads no values" measurement is a toy artefact as often as not,
        # so only a set the probe can close is checked from outside
        outside = sorted(
            r["name"] for k, r in by_key.items() if closed and k not in keys and seen(r)
        )
        line = f"{name}: {len(keys)} declared, {confirmed} confirmed by the probe"
        return line + (f"; seen outside it: {', '.join(outside)}" if outside else "")

    return [
        report(
            "POOLED_STATISTICS",
            tc.POOLED_STATISTICS,
            lambda r: r["time"] in ("pooled", "reduce"),
            closed=True,
        ),
        report(
            "TIME_INVARIANT",
            tc.TIME_INVARIANT,
            lambda r: r["measured"] == "reads no values",
            closed=False,
        ),
        report(
            "STACK_AS_FEATURES",
            tc.STACK_AS_FEATURES,
            lambda r: r["time"] == "reduce",
            closed=False,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write coupling.json and .md here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
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
