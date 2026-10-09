# Journal and resume

Restart a long patch job where it stopped. A
`geopatcher.observe.PatchJournal` is an append-only JSONL file with one
row per finished patch; pass it to `split(journal=...)` and anchors with
an `"ok"` row are skipped on the next run.

Use it for jobs over thousands of patches, on preemptible workers, or
when each patch writes an output you must not write twice. Skip it for
one-off exploration.

## Record progress and resume

Commit a row after each patch. A rerun with the same journal reads only
the anchors without an `"ok"` row.

```python
import time
from pathlib import Path

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
from geopatcher.observe import PatchJournal

Path("out/chips").mkdir(parents=True, exist_ok=True)
field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 256, 256), dtype=np.float32),       # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)


def run(journal: PatchJournal, crash_after: int | None = None) -> int:
    """Process every pending patch; optionally 'crash' after a few."""
    done: int = 0
    for patch in patcher.split(field, journal=journal):  # skips anchors with an "ok" row
        t0: float = time.perf_counter()
        out: np.ndarray = np.asarray(patch.data) * 2.0  # (1, 64, 64) float32
        uri: str = f"out/chips/{patch.anchor[0]}_{patch.anchor[1]}.npy"
        np.save(uri, out)
        journal.commit(patch.anchor, status="ok", runtime_s=time.perf_counter() - t0, output_uri=uri)
        done += 1
        if crash_after is not None and done == crash_after:
            break                                       # stand-in for a crash
    return done


first: int = run(PatchJournal("out/run.jsonl"), crash_after=5)  # 5 patches
journal: PatchJournal = PatchJournal("out/run.jsonl")            # reload after the "crash"
remaining: list[tuple[int, int]] = journal.pending(patcher.anchors(field))  # 11 anchors
second: int = run(journal)                                       # 11 patches, none repeated
```

`patcher.anchors(field)` lists the full anchor schedule without reading
any data. Record failures too, with `status="error"` and `error=...`;
they are retried on the next run because only `"ok"` rows count.

`geopatcher.run.parallel_map(..., journal=journal)` does this bookkeeping
for you, committing from the parent process — see
[Run in parallel with a journal](on-error-policies.md#run-in-parallel-with-a-journal).

## Journal methods

| Method | Behaviour |
|---|---|
| `journal.has(anchor)` | `True` if `anchor` has a `status == "ok"` row |
| `journal.commit(anchor, status=..., runtime_s=..., output_uri=None, error=None)` | append a row, then `flush()` and `fsync()` |
| `journal.pending(all_anchors)` | the anchors without an `"ok"` row |
| `journal.completed()` | every anchor whose latest row is `"ok"`, normalised |

## Anchor keys

Anchors are normalised by `geopatcher.observe.normalize_anchor` before
they are stored:

- numpy scalars become Python numbers; arrays and tuples become lists;
- `datetime64` / `timedelta64` become tagged dicts, so they never collide
  with a string anchor;
- anything else (an arbitrary object, a non-string dict key) raises
  `TypeError`.

So `(np.int64(5), np.int64(10))`, `np.array([5, 10])` and `(5, 10)` are the
same anchor, and `spatial.sampler.Explicit(anchors_=np.argwhere(mask))`
resumes correctly.

## Retries and the journal

The `on_error` policy handles transient errors within one run; the
journal carries progress across runs. A read that exhausts its retries
yields no patch, so it gets no `"ok"` row and the next run tries it
again. To give up on an anchor for good, commit an `"error"` row and
filter it out of your own schedule.

## Storage

- **Format.** One JSON object per line:
  `{"anchor": [0, 0], "status": "ok", "runtime_s": 0.31, "output_uri": "out/chips/0_0.npy", "error": null}`.
  `pandas.read_json(path, lines=True)` loads it for analysis.
- **Durability.** Each `commit` calls `fsync` before returning. Treat it
  as durable per row, not transactional: a truncated trailing row is
  dropped with a warning.
- **Location.** Local files only. Sync the file to object storage yourself
  if you need remote durability.
- **One writer per file.** The journal is not safe for several writers;
  give each worker its own file and merge them afterwards.
