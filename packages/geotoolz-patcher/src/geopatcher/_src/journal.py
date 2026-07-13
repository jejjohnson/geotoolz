"""Small local journal for resumable patch jobs."""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(eq=False)
class PatchJournal:
    """Append-only local journal keyed by patch anchor.

    Anchors are expected to be JSON-serializable tuples, lists, dictionaries,
    strings, numbers, or booleans. The journal stores one JSON record per
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
            "anchor": anchor,
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


def _anchor_key(anchor: Any) -> str:
    return json.dumps(anchor, sort_keys=True, default=str)
