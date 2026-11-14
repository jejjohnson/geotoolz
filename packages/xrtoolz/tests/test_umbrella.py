"""The umbrella holds no code (gate A2, #359)."""

from __future__ import annotations

import subprocess
import sys


def test_umbrella_exports_only_its_version() -> None:
    import xrtoolz

    assert xrtoolz.__all__ == ["__version__"]


def test_importing_the_umbrella_imports_no_ops_module() -> None:
    code = "import sys, xrtoolz; print(any(m.startswith('xrops') for m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
