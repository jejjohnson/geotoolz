# Observability hooks

`geopatcher` exposes a lightweight `PatcherHook` Protocol for progress bars,
tracing, metrics, and logging without adding hard dependencies. Pass hooks to
`split(..., hooks=[...])` or `merge(..., hooks=[...])`; each hook may implement
only the callbacks it needs.

## Callback order

For a complete split iteration:

```text
on_split_start(n_anchors)
  on_patch_start(anchor)
  on_patch_done(anchor, runtime_s, bytes_)
  ...
on_split_end()
```

For merge:

```text
on_merge_start(n_patches)
on_merge_end(output_bytes)
```

If patch construction or merging raises, `on_error(anchor, exc)` is called
before re-raising the original exception. Exceptions raised by hooks themselves
are converted to `RuntimeWarning`s so observability code cannot abort patching.
When a total is not cheaply knowable, patchers pass `-1`.

## `tqdm` progress bar

```python
from typing import Any

from tqdm import tqdm


class TqdmHook:
    def __init__(self) -> None:
        self.pbar = None

    def on_split_start(self, n_anchors: int) -> None:
        self.pbar = tqdm(total=None if n_anchors < 0 else n_anchors)

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        if self.pbar is not None:
            self.pbar.update(1)

    def on_split_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


for patch in patcher.split(field, hooks=[TqdmHook()]):
    ...
```

## OpenTelemetry tracing

```python
from typing import Any

from opentelemetry import trace


class OpenTelemetryHook:
    def __init__(self, tracer: trace.Tracer) -> None:
        self.tracer = tracer
        self.spans: dict[str, trace.Span] = {}

    def on_patch_start(self, anchor: Any) -> None:
        span = self.tracer.start_span("geopatcher.patch")
        span.set_attribute("geopatcher.anchor", repr(anchor))
        self.spans[repr(anchor)] = span

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        span = self.spans.pop(repr(anchor), None)
        if span is None:
            return
        span.set_attribute("geopatcher.runtime_s", runtime_s)
        span.set_attribute("geopatcher.bytes", bytes_)
        span.end()

    def on_error(self, anchor: Any, exc: Exception) -> None:
        span = self.spans.pop(repr(anchor), None)
        if span is not None:
            span.record_exception(exc)
            span.end()


tracer = trace.get_tracer(__name__)
for patch in patcher.split(field, hooks=[OpenTelemetryHook(tracer)]):
    ...
```
