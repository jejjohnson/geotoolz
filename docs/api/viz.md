# Viz

Display-time helpers: composites, stretches, colormaps, hillshade overlays. Outputs are typically
`uint8` RGB(A) for direct rendering.

- **Composites:** `Composite`, `TrueColor`, `FalseColor`, `SWIRComposite` (`spectral.SelectBands`
  presets: the selected bands' `band_names` / `wavelengths` travel with the output), and
  `RGBRecipe` — three band expressions with fixed `vmin` / `vmax` stretches and per-channel
  gamma, the form of the operational satellite RGBs (day cloud phase, fire temperature, …)
- **Stretches:** `StretchToUint8` (= `radiometry.PercentileClip` + a rounded byte cast; same
  `lower` / `upper` / `reduce_axes` arguments), `GammaCorrect` (radiometry's gamma on the display range)
- **Colormaps:** `ApplyColormap`, `ApplyDiscreteColormap`
- **Terrain:** `Hillshade`, `ShadedRelief`
- **Overlays:** `AnnotatePoints`, `AnnotatePolygons`, `Overlay`
- **Tier-A primitives:** `stretch_to_uint8`, `gamma_correct_display`, `rgb_recipe`, `rgba_from_scalar`,
  `rgba_from_categories`, `ensure_rgba`, `blend_rgba`

::: geotoolz.viz
