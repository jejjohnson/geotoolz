# Spatial axes

The four spatial axes composed by `SpatialPatcher`: Geometry ×
Sampler × Window × Aggregation (including the streaming sketch
aggregations).

## Spatial axes

### Geometry

::: geopatcher.SpatialGeometry
::: geopatcher.SpatialRectangular
::: geopatcher.SpatialSphericalCap
::: geopatcher.SpatialKNNGraph
::: geopatcher.SpatialRadiusGraph
::: geopatcher.SpatialPolygonIntersection

### Sampler

::: geopatcher.SpatialSampler
::: geopatcher.SpatialRegularStride
::: geopatcher.SpatialJitteredStride
::: geopatcher.SpatialRandom
::: geopatcher.SpatialPoissonDisk
::: geopatcher.SpatialExplicit
::: geopatcher.SpatialExplicitCoords
::: geopatcher.SpatialAlongTrack

### Window

::: geopatcher.SpatialWindow
::: geopatcher.SpatialBoxcar
::: geopatcher.SpatialHann
::: geopatcher.SpatialTukey
::: geopatcher.SpatialGaussian
::: geopatcher.SpatialCustom

### Aggregation

::: geopatcher.SpatialAggregation
::: geopatcher.SpatialSum
::: geopatcher.SpatialMean
::: geopatcher.SpatialVariance
::: geopatcher.SpatialOverlapAdd
::: geopatcher.SpatialWeightedSum
::: geopatcher.SpatialInvVarWeightedMean
::: geopatcher.SpatialMax
::: geopatcher.SpatialMin
::: geopatcher.SpatialMeanStd
::: geopatcher.SpatialMinMax
::: geopatcher.SpatialHardVote
::: geopatcher.SpatialSoftVote
::: geopatcher.SpatialByIndex
::: geopatcher.SpatialMedian
::: geopatcher.SpatialMode
::: geopatcher.SpatialLearned

#### Approximate (sketches)

::: geopatcher.SpatialApproxQuantile
::: geopatcher.SpatialApproxCardinality
::: geopatcher.SpatialApproxMode
::: geopatcher.SpatialStreamingHistogram
::: geopatcher.SpatialReservoir
