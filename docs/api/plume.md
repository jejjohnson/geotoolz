# Plume

Trace-gas retrieval post-processing for methane / CO₂ plumes — SBMP retrieval, detection masks,
footprint vectorisation, integrated mass enhancement, and cross-sectional flux. Algorithm sources
are cited in the operator docstrings (Varon et al., Frankenberg et al., Krings et al.).

- **Retrieval:** `SBMP` (Sentinel-2 SWIR ratio, Varon 2021)
- **Detection / segmentation:** `PlumeMask`, `PlumeContours`, `PlumeFootprint` (polygons from
  `georeader.vectorize.get_polygons`, `measure.DEFAULT_REGIONPROPS` metadata in pixel units)
- **Instance QA:** `PlumeShapeFilter` (fiber-shape morphology gate), `PlumeColumnStats`
  (in-mask vs background statistics), `PlumeQNDFeatures` (quantile-normality-deviation features)
- **Quantification:** `ColumnToMass`, `IMEEstimate`, `CrossSectionalFlux`, `WindAdvectionCone`
- **Tier-A primitives:** `plume_mask`, `convert_column_units`, `wind_advection_cone`

::: geotoolz.plume
