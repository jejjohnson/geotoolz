# ML utilities

Random-access views, patch stacking, and the operational-scale
primitives (journal, cache, backpressure) that ML loaders build on.

## Random access and stacking

::: geopatcher._src.indexed.IndexedPatchView
::: geopatcher._src.stacking.stack_patches

## Operational scale

::: geopatcher._src.journal.PatchJournal
::: geopatcher._src.journal.normalize_anchor
::: geopatcher.runners.parallel_map
::: geopatcher._src.prefetch.prefetch_iterable
