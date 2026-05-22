# Feature

`geotoolz.feature` wraps `skimage.feature` detectors as carrier-aware Operators. Inputs are
`GeoTensor`; coordinate outputs (peaks, blobs, corners) are returned as `GeoDataFrame`s in the input
CRS so they compose with the rest of the geospatial stack.

Operators in this module:

- **Edges / corners:** `Canny`, `CornerHarris`
- **Blobs:** `BlobDOG`, `BlobLoG`, `BlobDoH`
- **Local features:** `PeakLocalMax`, `StructureTensor`, `MultiscaleBasicFeatures`, `HOG`
- **Hough transforms:** `HoughLines`, `HoughCircles`

Operators that emit vector coordinates use `gt.transform` to project pixel `(row, col)` into world
`(x, y)` before constructing the output GeoDataFrame.

::: geotoolz.feature
