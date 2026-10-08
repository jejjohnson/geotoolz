"""`geopatcher.config` — saving and rebuilding patchers as plain data.

Every axis, stencil and patcher has ``get_config()``; `axis_envelope`
wraps one as ``{"class": <public path>, "config": {...}}`` (for example
``"spatial.window.Hann"``) and `from_config` rebuilds it — nested axes
included — after a JSON / YAML round-trip. `config_from_fields` is the
``get_config`` of a plain dataclass axis, for writing custom axes.
"""

from __future__ import annotations

from geopatcher._src._serialize import (
    axis_envelope,
    config_from_fields,
    config_name,
    from_config,
)


__all__ = ["axis_envelope", "config_from_fields", "config_name", "from_config"]
