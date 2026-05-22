# Plume

Trace-gas retrieval post-processing for methane / CO₂ plumes — SBMP retrieval, detection masks,
footprint vectorisation, integrated mass enhancement, and cross-sectional flux. Algorithm sources
are cited in the operator docstrings (Varon et al., Frankenberg et al., Krings et al.).

- **Retrieval:** `SBMP` (Sentinel-2 SWIR ratio, Varon 2021)
- **Detection / segmentation:** `PlumeMask`, `PlumeContours`, `PlumeFootprint` (with `regionprops`
  metadata in the output GDF)
- **Quantification:** `ColumnToMass`, `IMEEstimate`, `CrossSectionalFlux`, `WindAdvectionCone`

::: geotoolz.plume
