# Journal & resume — restartable patcher jobs

`PatchJournal` is a small append-only file that records `(anchor, status,
runtime, output_uri, error)` for each completed patch. Pass it to
`patcher.split(..., journal=journal)` and on restart the patcher skips
anchors that already have a successful row.

This recipe walks through:

1. The journal contract.
2. A minimal save / resume loop.
3. Combining a journal with `on_error="retry"`.
4. Storage notes (durability, format, file layout).

## When you need it

- Bulk inference on **thousands or millions of patches** where a single
  failure shouldn't restart the whole job.
- Multi-day jobs on spot instances / preemptible workers.
- Pipelines that write per-patch outputs to S3 / GCS as side-effects —
  the journal records *which writes already succeeded* so the rerun
  doesn't re-emit them.

For one-shot exploratory work, skip the journal.

## 1. The contract

```python
from geopatcher import PatchJournal

journal = PatchJournal("out/run.jsonl")
```

| Method | Behavior |
|---|---|
| `journal.has(anchor)` | `True` iff `anchor` has a `status == "ok"` row. |
| `journal.commit(anchor, status="ok", runtime_s=..., output_uri=..., error=None)` | Append a durable row. `flush()` + `fsync()` before return. |
| `journal.pending(all_anchors)` | Return the subset of `all_anchors` that don't have an `"ok"` row yet. |

The journal stores one JSON record per committed patch, keyed by the
JSON-serialised anchor. Anchors must be JSON-serialisable — tuples,
lists, dicts, strings, numbers, booleans all work; numpy scalars are
coerced via `default=str`.

**Durability.** Each `commit` flushes the Python buffer and calls
`os.fsync` on the file descriptor before returning. The OS may still
reorder the directory entry on power-loss, so treat the guarantee as
"best-effort durable per row" rather than transactional. Re-running a
job after a crash skips anchors with `status == "ok"`;
partially-written trailing rows are dropped by the JSON-decode guard
with a warning.

## 2. Minimal save / resume loop

```python
import dataclasses
import time

import numpy as np

from geopatcher import PatchJournal
import geopatcher as gp

journal = PatchJournal("out/lake-tahoe-run.jsonl")
patcher = gp.SpatialPatcher(
    geometry    = gp.SpatialRectangular(size=(256, 256)),
    sampler     = gp.SpatialRegularStride(step=(224, 224)),
    window      = gp.SpatialHann(),
    aggregation = gp.SpatialOverlapAdd(streaming=True, target_path="out/tahoe.zarr",
                                       chunks=(256, 256)),
)

for patch in patcher.split(field, journal=journal):
    # Skipped automatically if `journal.has(patch.anchor)` is True.
    t0 = time.perf_counter()
    try:
        out = my_operator(patch.data)
        out_uri = f"out/chips/{patch.anchor}.npy"
        np.save(out_uri, out)
        journal.commit(
            patch.anchor,
            status="ok",
            runtime_s=time.perf_counter() - t0,
            output_uri=out_uri,
        )
    except Exception as exc:
        journal.commit(
            patch.anchor,
            status="error",
            runtime_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
```

Kill the process at any point and rerun the same script — anchors with
an `"ok"` row are skipped on the next `patcher.split` call.

## 3. Combine with `on_error="retry"`

The journal and the `on_error` policy compose naturally. Use the journal
to record durable progress across restarts; use the policy to handle
transient I/O within a single run:

```python
patcher = gp.SpatialPatcher(
    ...,
    on_error    = "retry",
    max_retries = 3,
)

for patch in patcher.split(field, journal=journal):
    t0 = time.perf_counter()
    out = my_operator(patch.data)
    journal.commit(patch.anchor, status="ok", runtime_s=time.perf_counter() - t0)
```

If the read fails three times in a row, the retry policy logs to
`patcher.errors` and omits the anchor — the journal therefore never gets
an `"ok"` row, so the next restart re-tries the same anchor (perhaps
with the transient outage resolved).

If you instead want to mark exhausted-retry anchors as permanently
failed, walk `patcher.errors` after the loop and call
`journal.commit(anchor, status="error", ...)` explicitly so the next
restart skips them.

## 4. Restart story

On restart, the journal reloads its rows from the JSONL file. The
patcher walks the full anchor schedule and silently skips anchors that
have an `"ok"` row.

```python
journal = PatchJournal("out/lake-tahoe-run.jsonl")
print(f"already done: {sum(1 for k in journal._rows if journal._rows[k]['status'] == 'ok')}")

remaining = journal.pending(patcher.anchors(field))
print(f"remaining: {len(remaining)}")

for patch in patcher.split(field, journal=journal):
    ...
```

`patcher.anchors(field)` materialises the full anchor schedule without
reading the data — cheap relative to a real `split` walk.

## Storage notes

- **Format:** one JSON object per line (JSONL). Trivially diffable,
  grep-able, and consumable by `pandas.read_json(..., lines=True)` for
  ex-post analysis.
- **Location:** local filesystem. The journal does not write to S3 / GCS
  directly — wrap with `s3fs` or sync periodically if you need remote
  durability.
- **Concurrency:** the journal is *not* multi-writer safe. One process
  per journal file. For parallel runners, partition by journal file
  (one per worker) and post-process / merge.
- **Schema:**

  ```json
  {"anchor": [0, 0], "status": "ok", "runtime_s": 0.31,
   "output_uri": "out/chips/0_0.npy", "error": null}
  ```

## See also

- [`recipes/on-error-policies.md`](on-error-policies.md) — how transient I/O is retried inside a run.
- [`recipes/streaming-overlap-add.md`](streaming-overlap-add.md) — combine with disk-backed accumulation for resumable >1 TB outputs.
- [`observability.md`](../observability.md) — hook `on_patch_done` into the journal commit for tracing.
