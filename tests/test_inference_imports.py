"""D4 acceptance: importing :mod:`xr_toolz` (and even
:mod:`xr_toolz.inference`) must not pull ``sklearn``, ``jax``, or
``equinox`` into ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


_BANNED = ("sklearn", "jax", "jaxlib", "equinox", "torch")


def _leaked(script: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.strip().splitlines() if name]


def test_import_xr_toolz_does_not_leak_backends() -> None:
    leaked = _leaked(
        "import sys, xr_toolz\n"
        f"banned = {_BANNED!r}\n"
        "for k in sorted(sys.modules):\n"
        "    if k in banned or any(k.startswith(b + '.') for b in banned):\n"
        "        print(k)\n"
    )
    assert leaked == [], f"unexpected backend imports leaked: {leaked}"


def test_import_inference_does_not_leak_backends() -> None:
    leaked = _leaked(
        "import sys, xr_toolz.inference\n"
        f"banned = {_BANNED!r}\n"
        "for k in sorted(sys.modules):\n"
        "    if k in banned or any(k.startswith(b + '.') for b in banned):\n"
        "        print(k)\n"
    )
    assert leaked == [], f"unexpected backend imports leaked: {leaked}"


def test_modelop_does_not_export_from_root_package() -> None:
    """Per the lazy-backend rule, ``xr_toolz`` itself does not re-export
    ``ModelOp``; users opt in via ``xr_toolz.inference``.
    """
    import xr_toolz

    assert not hasattr(xr_toolz, "ModelOp")
    assert "ModelOp" not in xr_toolz.__all__


def test_inference_public_surface() -> None:
    import xr_toolz.inference as inference

    for name in ("ModelOp", "SklearnModelOp", "JaxModelOp"):
        assert hasattr(inference, name)
        assert name in inference.__all__


@pytest.mark.parametrize("backend", ["sklearn", "jax", "equinox", "torch"])
def test_modelop_module_source_does_not_top_level_import(backend: str) -> None:
    """Static check on the modelop source: no top-level
    ``import <backend>`` / ``from <backend> ...`` lines.
    """
    import xr_toolz.inference.modelop as mod

    with open(mod.__file__) as f:
        source = f.read()
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ")):
            indent = len(line) - len(stripped)
            if indent == 0:
                assert backend not in stripped, (
                    f"top-level import of {backend!r} found in modelop.py: {line!r}"
                )
