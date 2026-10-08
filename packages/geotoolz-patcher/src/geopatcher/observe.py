"""`geopatcher.observe` — watching and recovering a patching run.

- `PatcherHook` — the callback protocol ``split`` / ``merge`` notify
  (start, per-patch, error, end); `UNKNOWN_TOTAL` is the ``total``
  ``on_split_start`` receives when the patch count is not known up front.
- `PatchJournal` — a resumable record of the patches already processed
  (`normalize_anchor` is its anchor-key normaliser).
- `PatchErrorRecord` — what ``on_error="skip" | "mask"`` records for a
  failed patch; `OnErrorPolicy` names the policies.
- `get_strict` / `set_strict` — the process-wide strict mode that turns
  silent fallbacks into errors.
"""

from __future__ import annotations

from geopatcher._src.config import get_strict, set_strict
from geopatcher._src.hooks import UNKNOWN_TOTAL, PatcherHook
from geopatcher._src.journal import PatchJournal, normalize_anchor
from geopatcher._src.walk import OnErrorPolicy, PatchErrorRecord


__all__ = [
    "UNKNOWN_TOTAL",
    "OnErrorPolicy",
    "PatchErrorRecord",
    "PatchJournal",
    "PatcherHook",
    "get_strict",
    "normalize_anchor",
    "set_strict",
]
