# Config round-trips — `geopatcher.config`

Every axis, stencil and patcher exposes `get_config()`. Nested components
serialise as `{"class": ..., "config": ...}` envelopes whose `"class"` is
the component's public path — `"spatial.window.Hann"`,
`"temporal.aggregation.Mean"`, `"SpatialPatcher"` (a user-defined axis
records its `module.qualname`). `from_config(axis_envelope(obj))` rebuilds
`obj`, unless its type is `forbid_in_yaml` (closures, polygons,
backend-native anchors), whose config is a debug summary `from_config`
refuses.

```python
import json

from geopatcher import spatial
from geopatcher.config import axis_envelope, from_config

window = spatial.window.Tukey(alpha=0.25)
saved = json.dumps(axis_envelope(window))      # '{"class": "spatial.window.Tukey", ...}'
assert from_config(json.loads(saved)).get_config() == window.get_config()
```

::: geopatcher.config.axis_envelope
::: geopatcher.config.from_config
::: geopatcher.config.config_name
::: geopatcher.config.config_from_fields
