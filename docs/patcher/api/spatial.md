# Spatial axes — `geopatcher.spatial`

The four spatial axes composed by `SpatialPatcher`: Geometry ×
Sampler × Window × Aggregation (including the streaming sketch
aggregations).

## Spatial axes

### Geometry

::: geopatcher.spatial.geometry.Geometry
::: geopatcher.spatial.geometry.Rectangular
::: geopatcher.spatial.geometry.SphericalCap
::: geopatcher.spatial.geometry.KNNGraph
::: geopatcher.spatial.geometry.RadiusGraph
::: geopatcher.spatial.geometry.PolygonIntersection

### Sampler

::: geopatcher.spatial.sampler.Sampler
::: geopatcher.spatial.sampler.RegularStride
::: geopatcher.spatial.sampler.JitteredStride
::: geopatcher.spatial.sampler.Random
::: geopatcher.spatial.sampler.PoissonDisk
::: geopatcher.spatial.sampler.Explicit
::: geopatcher.spatial.sampler.ExplicitCoords
::: geopatcher.spatial.sampler.AlongTrack
::: geopatcher.spatial.sampler.IncompleteScanConfiguration

### Window

::: geopatcher.spatial.window.Window
::: geopatcher.spatial.window.Boxcar
::: geopatcher.spatial.window.Hann
::: geopatcher.spatial.window.Tukey
::: geopatcher.spatial.window.Gaussian
::: geopatcher.spatial.window.Custom
::: geopatcher.spatial.window.geom_shape

### Aggregation

::: geopatcher.spatial.aggregation.Aggregation
::: geopatcher.spatial.aggregation.Sum
::: geopatcher.spatial.aggregation.Mean
::: geopatcher.spatial.aggregation.Variance
::: geopatcher.spatial.aggregation.OverlapAdd
::: geopatcher.spatial.aggregation.WeightedSum
::: geopatcher.spatial.aggregation.InvVarWeightedMean
::: geopatcher.spatial.aggregation.Max
::: geopatcher.spatial.aggregation.Min
::: geopatcher.spatial.aggregation.MeanStd
::: geopatcher.spatial.aggregation.MinMax
::: geopatcher.spatial.aggregation.HardVote
::: geopatcher.spatial.aggregation.SoftVote
::: geopatcher.spatial.aggregation.ByIndex
::: geopatcher.spatial.aggregation.Median
::: geopatcher.spatial.aggregation.Mode
::: geopatcher.spatial.aggregation.Learned

#### Approximate (sketches)

::: geopatcher.spatial.aggregation.ApproxQuantile
::: geopatcher.spatial.aggregation.ApproxCardinality
::: geopatcher.spatial.aggregation.ApproxMode
::: geopatcher.spatial.aggregation.StreamingHistogram
::: geopatcher.spatial.aggregation.Reservoir
