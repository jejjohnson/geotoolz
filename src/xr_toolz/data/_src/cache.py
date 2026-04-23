"""Tiny on-disk cache keyed on request parameters.

Adapters call :func:`cache_path` to get a deterministic path for a
given ``(source, dataset_id, request)`` triple. If the path exists,
they can short-circuit the network call; if it doesn't, they download
into it and the filesystem becomes the cache.

The cache root defaults to ``$XR_TOOLZ_CACHE`` or
``~/.cache/xr_toolz/data``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def cache_root() -> Path:
    """Return the root cache directory, creating it if needed."""
    env = os.environ.get("XR_TOOLZ_CACHE")
    root = Path(env) if env else Path.home() / ".cache" / "xr_toolz" / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_path(
    source: str,
    dataset_id: str,
    request: dict[str, Any],
    suffix: str = ".nc",
) -> Path:
    """Return a deterministic cache path for ``(source, dataset_id, request)``.

    The path is ``<root>/<source>/<dataset_id>/<request_hash><suffix>``.
    """
    base = cache_root() / source / _safe(dataset_id)
    base.mkdir(parents=True, exist_ok=True)
    digest = _hash_request(request)
    return base / f"{digest}{suffix}"


def _hash_request(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        # Sets iterate in hash-randomized order across interpreter runs;
        # sort so the cache key is stable run-to-run.
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _safe(name: str) -> str:
    """Make ``name`` safe to use as a path component."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
