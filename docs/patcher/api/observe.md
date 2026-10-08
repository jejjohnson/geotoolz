# Observing a run — `geopatcher.observe`

Callbacks during `split` / `merge`, a resumable record of the processed
patches, what a failed patch leaves behind, and strict mode. The
[observability guide](../observability.md) and the
[on-error](../recipes/on-error-policies.md) /
[journal](../recipes/journal-and-resume.md) recipes show them in use.

```python
from geopatcher.observe import PatchJournal, PatcherHook
```

## Hooks

::: geopatcher.observe.PatcherHook
::: geopatcher.observe.UNKNOWN_TOTAL

## Journal

::: geopatcher.observe.PatchJournal
::: geopatcher.observe.normalize_anchor

## Errors

::: geopatcher.observe.PatchErrorRecord

## Strict mode

::: geopatcher.observe.get_strict
::: geopatcher.observe.set_strict
