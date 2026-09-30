# Viz

Display-time helpers: composites, stretches, colormaps, hillshade overlays. Outputs are typically
`uint8` RGB(A) for direct rendering.

- **Composites:** `Composite`, `TrueColor`, `FalseColor`, `SWIRComposite` (`spectral.SelectBands`
  presets: the selected bands' `band_names` / `wavelengths` travel with the output)
- **Stretches:** `StretchToUint8` (= `radiometry.PercentileClip` + a rounded byte cast; same
  `lower` / `upper` / `axis` arguments), `GammaCorrect` (radiometry's gamma on the display range)
- **Colormaps:** `ApplyColormap`, `ApplyDiscreteColormap`
- **Terrain:** `Hillshade`, `ShadedRelief`
- **Overlays:** `AnnotatePoints`, `AnnotatePolygons`, `Overlay`

::: geotoolz.viz
