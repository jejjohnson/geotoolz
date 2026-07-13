# Async and prefetch patching

`SpatialPatcher.split(field)` stays synchronous by default. When reads are slower than the operator, pass `prefetch=N` to overlap patch reads with downstream work on a background thread:

```python
patches = []
for patch in patcher.split(field, prefetch=4):
    out = heavy_operator(patch.data)
    patches.append(type(patch)(data=out, anchor=patch.anchor, indices=patch.indices, weights=patch.weights))

stitched = patcher.merge(patches, field.domain)
```

Use `prefetch=0` for the original serial path. Worker exceptions are re-raised on the consumer thread, so normal `try` / `except` handling around the loop still works.

For cloud-hosted readers that expose async I/O, use `asplit()` with an `AsyncField`:

```python
async def run(async_field):
    patches = []
    async for patch in patcher.asplit(async_field):
        out = await async_operator(patch.data)
        patches.append(type(patch)(data=out, anchor=patch.anchor, indices=patch.indices, weights=patch.weights))

    return await patcher.amerge(patches, async_field.domain)
```

`AsyncRasterField` supports both `select()` and `aselect()` coroutine names, and custom fields can implement either name. The patcher reads one patch at a time and leaves concurrency policy for the operator to the caller, which keeps memory bounded and avoids imposing an event-loop backend.
