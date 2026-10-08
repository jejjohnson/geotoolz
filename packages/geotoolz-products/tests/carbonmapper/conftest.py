"""CarbonMapper tests need the ``[carbonmapper]`` extra (requests, pydantic).

On a slim install the whole directory is left uncollected — the package's
import guard is exercised by ``test_carbonmapper_extra.py`` instead.
"""

from __future__ import annotations

import importlib.util


collect_ignore_glob = (
    ["test_*.py"]
    if any(importlib.util.find_spec(m) is None for m in ("requests", "pydantic"))
    else []
)
