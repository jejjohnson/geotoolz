# Viz

Display-time helpers: composites, stretches, colormaps, hillshade overlays. Outputs are typically
`uint8` RGB(A) for direct rendering.

- **Composites:** `Composite`, `TrueColor`, `FalseColor`, `SWIRComposite`
- **Stretches:** `StretchToUint8`, `ToDisplayRange`, `GammaCorrect`
- **Colormaps:** `ApplyColormap`, `ApplyDiscreteColormap`
- **Terrain:** `Hillshade`, `ShadedRelief`
- **Overlays:** `AnnotatePoints`, `AnnotatePolygons`, `Overlay`

::: geotoolz.viz
