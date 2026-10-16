"""Optional bridges from `geopatcher` into downstream composition libraries.

Each submodule is gated behind an optional extra so the geopatcher core
install stays slim:

- `geopatcher.integrations.pipekit` — Operator wrappers for the
  pipekit operator-graph framework. Install with the `[pipekit]` extra.
- `geopatcher.integrations.train` — `XarrayWindowDataset`, a pipekit-train
  `TrainingDataset` of windows from lazy xarray / Zarr stores. Install with
  the `[train]` extra.
"""
