"""The committed carrier-versus-xarray benchmark runs (gate GA-G22, #536)."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "carrier_vs_xarray.py"


def test_benchmark_runs_on_the_smallest_shape(tmp_path: Path) -> None:
    """The script runs on the smallest shape in under 10 s and writes its JSON."""
    pytest.importorskip("xarray")
    spec = importlib.util.spec_from_file_location("carrier_vs_xarray", SCRIPT)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    out = tmp_path / "result.json"
    start = time.perf_counter()
    bench.main(["--smallest", "--repeats", "1", "--out", str(out)])
    assert time.perf_counter() - start < 10.0
    result = json.loads(out.read_text())
    (row,) = result["rows"]
    assert row["shape"] == [2, 64, 64]
    assert all(row[key] > 0 for key in row if key.endswith("_us"))
    # xarray's inner join drops every pixel; the carrier refuses
    assert result["misaligned_1e-7_m"]["xarray_result_shape"] == [1, 4, 0]
    assert result["misaligned_1e-7_m"]["carrier"].startswith("GridMismatchError")
