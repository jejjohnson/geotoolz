"""`PatchCache` — content-addressed, on-disk patch cache (gh #24).

Cross-run sibling of `IndexedPatchView(cache=True)` (which only avoids
re-reads *within* a process). A `PatchCache` keys each patch by

    sha256( field_id ‖ config_id ‖ anchor_id )

so a second *process* — a rerun after editing the operator — skips the
source read entirely and consults the field only for its ``domain``
metadata.

- ``field_id`` — stable identity of the source (see `PatchCache.field_id_for`).
- ``config_id`` — ``json.dumps`` of the geometry + window configs
  (sampler / aggregation excluded: they don't change patch bytes).
- ``anchor_id`` — `PatchJournal`'s anchor normaliser, reused verbatim.

Each entry is one ``<hash>.npz`` holding ``values`` (+ ``transform`` /
``crs`` when the carrier is georeferenced, ``weights`` when present).
`Patch.anchor` / `Patch.indices` are recomputed by the patcher, not
stored. Writes are atomic (``.tmp`` + ``os.replace``); eviction is LRU
by file ``st_atime`` when ``max_bytes`` is exceeded, checked on write.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geopatcher._src.journal import _anchor_key
from geopatcher._src.patch import Patch


@dataclass
class PatchCache:
    """A local, content-addressed directory cache of read patches.

    Args:
        root: Directory the cache writes entries under (created if
            absent). The layout is fsspec-friendly (two-level shard) but
            v1 is a local filesystem only.
        max_bytes: Soft cap on the cache size. When exceeded after a
            write, least-recently-accessed entries are evicted until the
            total is back under the cap. ``None`` (default) is unbounded.
        field_id: Explicit source identity for in-memory fields that
            have none (a bare `GeoTensor`-backed `RasterField`). Leave
            ``None`` for path- or URL-backed fields, whose identity is
            derived automatically.
    """

    root: str | Path
    max_bytes: int | None = None
    field_id: str | None = None
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ValueError("max_bytes must be >= 1 (or None for unbounded).")

    # -- key derivation ---------------------------------------------------

    def field_id_for(self, field: Any) -> str:
        """Resolve a stable source identity for ``field``.

        Resolution order: an explicit ``field_id`` on this cache, then a
        field-provided ``cache_id()`` escape hatch, then an
        `ObstoreCogField`-style ``url``, then a reader with a filesystem
        path (``realpath`` + ``st_mtime_ns`` + ``st_size``). In-memory
        fields with none of these raise, since caching against nothing
        would silently serve stale data.
        """
        if self.field_id is not None:
            return self.field_id
        cache_id = getattr(field, "cache_id", None)
        if callable(cache_id):
            return f"id:{cache_id()}"
        url = getattr(field, "url", None)
        if url is not None:
            return f"url:{url}"
        reader = getattr(field, "reader", None)
        for obj in (field, reader):
            path = getattr(obj, "paths", None) or getattr(obj, "path", None)
            if not path:
                continue
            if isinstance(path, (list, tuple)):
                path = path[0]
            with suppress(OSError, TypeError):
                st = os.stat(path)
                return f"path:{os.path.realpath(path)}:{st.st_mtime_ns}:{st.st_size}"
        raise ValueError(
            f"PatchCache cannot derive a stable identity for a "
            f"{type(field).__name__}: it exposes no cache_id(), url, or file "
            f"path. Pass field_id=... to PatchCache for in-memory fields."
        )

    @staticmethod
    def config_id_for(geometry: Any, window: Any) -> str:
        """JSON config key from the geometry + window (byte-affecting axes)."""
        for axis in (geometry, window):
            if getattr(axis, "forbid_in_yaml", False):
                raise ValueError(
                    f"{type(axis).__name__} carries a Python closure and cannot "
                    f"be used as a cache key; give it a serialisable config or "
                    f"drop the cache for this run."
                )
        return json.dumps(
            {"geometry": geometry.get_config(), "window": window.get_config()},
            sort_keys=True,
        )

    def _key(self, field_id: str, config_id: str, anchor: Any) -> str:
        h = hashlib.sha256()
        h.update(field_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(config_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(_anchor_key(anchor).encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return Path(self.root) / key[:2] / f"{key}.npz"

    # -- get / put --------------------------------------------------------

    def get(self, field_id: str, config_id: str, anchor: Any) -> dict[str, Any] | None:
        """Return the stored payload for a key, or ``None`` on a miss."""
        path = self._path(self._key(field_id, config_id, anchor))
        if not path.exists():
            self._misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                payload = {k: npz[k] for k in npz.files}
        except (OSError, ValueError, EOFError):
            # A truncated / partially-written entry: treat as a miss.
            self._misses += 1
            return None
        self._hits += 1
        if self.max_bytes is not None:
            # Refresh the access time so eviction sees true recency even on
            # relatime/noatime mounts.
            with suppress(OSError):
                os.utime(path)
        return payload

    def put(self, field_id: str, config_id: str, anchor: Any, patch: Patch) -> None:
        """Store ``patch``'s reconstructable payload under its key."""
        path = self._path(self._key(field_id, config_id, anchor))
        if path.exists():
            return
        values, transform, crs, weights = _extract_payload(patch)
        arrays: dict[str, Any] = {
            "values": values,
            "has_geo": np.array(transform is not None),
            "transform": _affine_floats(transform),
            "crs": np.array("" if crs is None else str(crs)),
            "has_weights": np.array(weights is not None),
        }
        if weights is not None:
            arrays["weights"] = np.asarray(weights)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp per writer (mkstemp is atomic) so concurrent writers
        # of the same key — same PID across threads included — never share
        # a scratch file; the os.replace then publishes atomically.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
                f.flush()
                with suppress(OSError):
                    os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            with suppress(OSError):
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        if self.max_bytes is not None:
            self._evict()

    def build_patch(self, payload: dict[str, Any], anchor: Any, indices: Any) -> Patch:
        """Rebuild a `Patch` from a stored ``payload`` at ``anchor``/``indices``."""
        values = payload["values"]
        if bool(payload["has_geo"]):
            from georeader.geotensor import GeoTensor
            from rasterio import Affine

            crs = str(payload["crs"])
            data: Any = GeoTensor(
                values=values,
                transform=Affine(*payload["transform"].tolist()),
                crs=crs or None,
            )
        else:
            data = values
        weights = payload["weights"] if bool(payload["has_weights"]) else None
        return Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    # -- introspection ----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return ``{"hits", "misses", "bytes", "entries"}`` for the cache."""
        total = 0
        entries = 0
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                total += path.stat().st_size
                entries += 1
        return {
            "hits": self._hits,
            "misses": self._misses,
            "bytes": total,
            "entries": entries,
        }

    def clear(self) -> None:
        """Delete every cached entry; reset hit / miss counters."""
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                path.unlink()
        self._hits = 0
        self._misses = 0

    def _evict(self) -> None:
        """Drop least-recently-accessed entries until under ``max_bytes``."""
        if self.max_bytes is None:
            return
        entries: list[tuple[float, int, Path]] = []
        for path in self.root.rglob("*.npz"):
            with suppress(OSError):
                st = path.stat()
                entries.append((st.st_atime, st.st_size, path))
        total = sum(size for _, size, _ in entries)
        if total <= self.max_bytes:
            return
        entries.sort()  # oldest access time first
        for _atime, size, path in entries:
            if total <= self.max_bytes:
                break
            with suppress(OSError):
                path.unlink()
                total -= size


def _extract_payload(patch: Patch) -> tuple[np.ndarray, Any, Any, Any]:
    """Pull ``(values, transform, crs, weights)`` out of a built patch."""
    data = patch.data
    da = getattr(data, "da", None)
    if da is not None:  # xarray-backed carrier (RioXarrayField, …)
        values = np.asarray(da.values)
    else:
        values = np.asarray(getattr(data, "values", data))
    transform = getattr(data, "transform", None)
    crs = getattr(data, "crs", None)
    return values, transform, crs, patch.weights


def _affine_floats(transform: Any) -> np.ndarray:
    """Six-float representation of a rasterio ``Affine`` (zeros when absent)."""
    if transform is None:
        return np.zeros(6, dtype=float)
    return np.array(
        [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        dtype=float,
    )


__all__ = ["PatchCache"]
