"""Small local journal for resumable patch jobs."""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(eq=False)
class PatchJournal:
    """Append-only local journal keyed by patch anchor.

    Anchors may be tuples, lists, dictionaries (string keys), strings,
    numbers, booleans, numpy scalars or numpy arrays — everything the
    samplers emit, including ``SpatialExplicit(anchors_=np.argwhere(...))``
    rows. Every anchor goes through `normalize_anchor` before it is keyed
    or written, so ``(np.int64(5), np.int64(10))``, ``np.array([5, 10])``
    and ``(5, 10)`` are one anchor. The journal stores one JSON record per
    committed patch. Re-opening the same path reconstructs the latest status
    for each anchor, allowing ``patcher.split(..., journal=journal)`` to skip
    completed work after a crash.

    Durability: each ``commit`` flushes the Python buffer and calls
    ``os.fsync`` on the file descriptor before returning. The OS may still
    reorder the directory entry on a power-loss event, so treat the
    guarantee as "best-effort durable per row" rather than transactional.
    Re-running a job after a crash skips anchors that have a row with
    ``status == "ok"``; partially-written trailing rows are dropped by the
    JSON-decode guard in ``_load`` with a warning.
    """

    uri: str

    def __post_init__(self) -> None:
        self.path = Path(self.uri)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def has(self, anchor: Any) -> bool:
        """Return ``True`` when ``anchor`` has a successful journal row."""
        row = self._rows.get(_anchor_key(anchor))
        return row is not None and row["status"] == "ok"

    def commit(
        self,
        anchor: Any,
        *,
        status: str,
        runtime_s: float,
        output_uri: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append a durable status row for ``anchor``.

        The row is flushed and ``fsync``-ed before the call returns so a
        process crash after ``commit()`` returns does not lose the record.
        """
        row = {
            "anchor": normalize_anchor(anchor),
            "status": status,
            "runtime_s": float(runtime_s),
            "output_uri": output_uri,
            "error": error,
        }
        key = _anchor_key(anchor)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
            f.flush()
            # fsync isn't supported on every filesystem (e.g. some network
            # mounts, tmpfs in container CI). Fall back to the plain flush
            # in that case — the row is still in the OS page cache.
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        self._rows[key] = row

    def pending(self, all_anchors: list[Any]) -> list[Any]:
        """Return anchors without a successful journal row."""
        return [anchor for anchor in all_anchors if not self.has(anchor)]

    def completed(self) -> list[Any]:
        """Return every anchor whose latest row has ``status == "ok"``.

        Anchors come back in their journal (JSON-normalised) form — see
        `normalize_anchor`: tuples and numpy arrays as lists, numpy
        scalars as Python numbers.

        Examples:
            >>> journal = PatchJournal("out/run.jsonl")
            >>> journal.commit((0, 0), status="ok", runtime_s=0.1)
            >>> journal.commit((0, 8), status="error", runtime_s=0.1)
            >>> journal.completed()
            [[0, 0]]
            >>> len(journal.completed())  # progress on restart
            1
        """
        return [row["anchor"] for row in self._rows.values() if row["status"] == "ok"]

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    warnings.warn(
                        f"skipping malformed journal row in {self.path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                self._rows[_anchor_key(row["anchor"])] = row


def normalize_anchor(anchor: Any) -> Any:
    """Return ``anchor`` as plain JSON-compatible Python values.

    The one anchor normaliser shared by `PatchJournal` (rows and keys)
    and `PatchCache` (entry keys), so an anchor is the same key however
    a sampler spelled it:

    - numpy scalars → ``.item()`` (``datetime64`` / ``timedelta64`` →
      ``str``, so they stay JSON-serialisable);
    - numpy arrays → nested lists of normalised elements;
    - tuples and lists → lists;
    - dicts → dicts of normalised values (keys must be strings).

    Args:
        anchor: A sampler anchor.

    Returns:
        The normalised anchor.

    Raises:
        TypeError: ``anchor`` contains a value with no JSON form (an
            arbitrary object, a non-string dict key). There is no
            ``str()`` fallback: two distinct objects with the same
            ``repr`` would otherwise collide.

    Examples:
        >>> normalize_anchor((np.int64(5), np.int64(10)))
        [5, 10]
        >>> normalize_anchor(np.array([41, 31]))
        [41, 31]
        >>> normalize_anchor({"time": np.int32(2), "lat": 0})
        {'time': 2, 'lat': 0}
    """
    if isinstance(anchor, np.ndarray):
        if anchor.ndim == 0:
            return normalize_anchor(anchor[()])
        return [normalize_anchor(v) for v in anchor]
    if isinstance(anchor, (np.datetime64, np.timedelta64)):
        return str(anchor)
    if isinstance(anchor, np.generic):
        return anchor.item()
    if isinstance(anchor, (tuple, list)):
        return [normalize_anchor(v) for v in anchor]
    if isinstance(anchor, dict):
        if not all(isinstance(k, str) for k in anchor):
            raise TypeError(
                f"anchor dict keys must be strings to be journalled / cached; "
                f"got {anchor!r}"
            )
        return {k: normalize_anchor(v) for k, v in anchor.items()}
    if anchor is None or isinstance(anchor, (bool, int, float, str)):
        return anchor
    raise TypeError(
        f"cannot journal / cache an anchor containing a "
        f"{type(anchor).__qualname__} ({anchor!r}); anchors must be numbers, "
        f"strings, booleans, numpy scalars / arrays, or tuples / lists / "
        f"string-keyed dicts of them."
    )


def _anchor_key(anchor: Any) -> str:
    """Canonical JSON key of ``anchor`` (via `normalize_anchor`)."""
    return json.dumps(normalize_anchor(anchor), sort_keys=True)
