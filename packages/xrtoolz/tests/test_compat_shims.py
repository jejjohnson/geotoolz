"""Guards for the post-split backward-compatibility surface.

The workspace split moved ``calc`` → ``xrgrad``, ``einx`` → ``xreinx``,
the sklearn bridge → ``xrsklearn``, and ``Operator`` / ``Signature`` →
``xrcore``. These tests pin the promises the split made:

- moved modules stay importable from their old paths (with a
  ``DeprecationWarning`` for the renamed distributions),
- old attribute homes (``xrtoolz.utils.XarrayEstimator``,
  ``xrtoolz.transforms.SklearnOp``, top-level ``Operator`` /
  ``Signature``) keep resolving to the same objects,
- ``import xrtoolz`` stays light: no einx, no JAX.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_calc_shim_warns_and_reexports() -> None:
    with pytest.warns(DeprecationWarning, match="xrtoolz-grad"):
        import xrtoolz.calc as shim

    import xrgrad

    assert shim.partial is xrgrad.partial
    assert shim.GRAVITY == xrgrad.GRAVITY


def test_einx_shim_warns_and_reexports() -> None:
    with pytest.warns(DeprecationWarning, match="xrtoolz-einx"):
        import xrtoolz.einx as shim

    import xreinx

    assert shim.einsum is xreinx.einsum
    assert shim.Einsum is xreinx.Einsum


def test_einx_shim_keeps_public_submodule_paths() -> None:
    import xreinx
    from xrtoolz.einx.dataset import pack_dataset
    from xrtoolz.einx.operators import Einsum

    assert pack_dataset is xreinx.pack_dataset
    assert Einsum is xreinx.Einsum


def test_core_and_sklearn_names_keep_resolving() -> None:
    import xrcore
    import xrsklearn
    import xrtoolz
    import xrtoolz.signature
    from xrtoolz.transforms import SklearnOp
    from xrtoolz.utils import XarrayEstimator

    assert xrtoolz.Operator is xrcore.Operator
    assert xrtoolz.Signature is xrcore.Signature
    assert xrtoolz.signature.Signature is xrcore.Signature
    assert XarrayEstimator is xrsklearn.XarrayEstimator
    assert SklearnOp is xrsklearn.SklearnOp


@pytest.mark.slow
def test_import_xrtoolz_stays_light() -> None:
    code = (
        "import sys, xrtoolz; "
        "assert 'einx' not in sys.modules, 'import xrtoolz pulled in einx'; "
        "assert 'jax' not in sys.modules, 'import xrtoolz pulled in jax'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
