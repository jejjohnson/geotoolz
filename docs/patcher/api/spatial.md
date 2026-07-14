# Spatial axes

The four spatial axes composed by `SpatialPatcher`: Geometry ×
Sampler × Window × Aggregation (including the streaming sketch
aggregations).

## Spatial axes

### Geometry

::: geopatcher._src.spatial.geometry.SpatialGeometry
::: geopatcher._src.spatial.geometry.SpatialRectangular
::: geopatcher._src.spatial.geometry.SpatialSphericalCap
::: geopatcher._src.spatial.geometry.SpatialKNNGraph
::: geopatcher._src.spatial.geometry.SpatialRadiusGraph
::: geopatcher._src.spatial.geometry.SpatialPolygonIntersection

### Sampler

::: geopatcher._src.spatial.sampler.SpatialSampler
::: geopatcher._src.spatial.sampler.SpatialRegularStride
::: geopatcher._src.spatial.sampler.SpatialJitteredStride
::: geopatcher._src.spatial.sampler.SpatialRandom
::: geopatcher._src.spatial.sampler.SpatialPoissonDisk
::: geopatcher._src.spatial.sampler.SpatialExplicit
::: geopatcher._src.spatial.sampler.SpatialAlongTrack

### Window

::: geopatcher._src.spatial.window.SpatialWindow
::: geopatcher._src.spatial.window.SpatialBoxcar
::: geopatcher._src.spatial.window.SpatialHann
::: geopatcher._src.spatial.window.SpatialTukey
::: geopatcher._src.spatial.window.SpatialGaussian
::: geopatcher._src.spatial.window.SpatialCustom

### Aggregation

::: geopatcher._src.spatial.aggregation.SpatialAggregation
::: geopatcher._src.spatial.aggregation.SpatialSum
::: geopatcher._src.spatial.aggregation.SpatialMean
::: geopatcher._src.spatial.aggregation.SpatialVariance
::: geopatcher._src.spatial.aggregation.SpatialOverlapAdd
::: geopatcher._src.spatial.aggregation.SpatialWeightedSum
::: geopatcher._src.spatial.aggregation.SpatialInvVarWeightedMean
::: geopatcher._src.spatial.aggregation.SpatialMax
::: geopatcher._src.spatial.aggregation.SpatialMin
::: geopatcher._src.spatial.aggregation.SpatialMeanStd
::: geopatcher._src.spatial.aggregation.SpatialMinMax
::: geopatcher._src.spatial.aggregation.SpatialHardVote
::: geopatcher._src.spatial.aggregation.SpatialSoftVote
::: geopatcher._src.spatial.aggregation.SpatialByIndex
::: geopatcher._src.spatial.aggregation.SpatialMedian
::: geopatcher._src.spatial.aggregation.SpatialMode
::: geopatcher._src.spatial.aggregation.SpatialLearned

#### Approximate (sketches)

::: geopatcher._src.spatial.aggregation.SpatialApproxQuantile
::: geopatcher._src.spatial.aggregation.SpatialApproxCardinality
::: geopatcher._src.spatial.aggregation.SpatialApproxMode
::: geopatcher._src.spatial.aggregation.SpatialStreamingHistogram
::: geopatcher._src.spatial.aggregation.SpatialReservoir
