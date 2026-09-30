# Compositing

Per-pixel composites over co-registered frame stacks and matched multi-source tuples.

- **Temporal composites:** `MedianComposite`, `MaxNDVIComposite` (NDVI via
  `geotoolz.indices.ndvi`), `CloudFreeComposite`, `MinCloudComposite`, `BAPComposite`
- **Multi-source fusion:** `StackMatched`, `BlendMatched`
- **Tier-A primitives** (plain `(T, ..., H, W)` numpy stacks, NaN = missing): `median_composite`,
  `mean_composite`, `take_by_spatial_index`, `mask_frames`, `blend_weighted`, `bap_scores` and the
  per-criterion BAP scores `doy_distance`, `doy_score`, `view_angle_score`, `cloud_distance_score`,
  `opacity_score`

::: geotoolz.compositing
