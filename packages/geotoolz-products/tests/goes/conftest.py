"""Fixtures for the GOES-R ABI tests.

The reader tests need the ``[goes]`` extra (h5py); on a slim install they
are left uncollected and only the ``aws`` tests (which skip without
geotoolz-cloud) and the preset tests run.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


collect_ignore_glob = (
    [
        "test_goes_reader.py",
        "test_goes_l2.py",
        "test_goes_stack.py",
        "test_goes_integration.py",
    ]
    if importlib.util.find_spec("h5py") is None
    else []
)


@pytest.fixture
def abi_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``abi_file(**write_abi_kwargs)`` → path of a synthetic file."""
    counter = iter(range(1_000))

    def make(**kwargs: Any) -> Path:
        from _goes_abi import write_abi

        return write_abi(tmp_path / f"abi_{next(counter)}.nc", **kwargs)

    return make


@pytest.fixture
def l2_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``l2_file(variables, **write_l2_kwargs)`` → path."""
    counter = iter(range(1_000))

    def make(variables: dict[str, Any], **kwargs: Any) -> Path:
        from _goes_abi import write_l2

        return write_l2(tmp_path / f"l2_{next(counter)}.nc", variables, **kwargs)

    return make
