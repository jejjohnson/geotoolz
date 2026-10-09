# IO

Reader and writer source/sink operators. For HDF / HDF-EOS / NetCDF-CF
reads and their extras, see [Read HDF and NetCDF files](../io.md).

- **Window / bounds readers:** `ReadWindow`, `ReadBounds`, `ReadCenterCoords`, `ReadPolygon`,
  `ReadTile`, `ReadToCRS`, `ReadReprojectLike`
- **Multi-format:** `ReadHDF` (HDF5/HDF4 dispatch), `ReadNetCDF` (CF mask/scale + group nav)
- **Cloud / catalog source:** `LoadFromEE`, `LoadFromSTAC`
- **Writers:** `WriteGeoTIFF`, `WriteCOG`, `WriteZarr`
- **Base classes:** `SourceOperator`, `SinkOperator`, `GeoToolzIOError`
- **Tier-A primitives:** `select_indexes` / `read_indexes` (1-based band selection),
  `fill_value_from_attrs` (CF / HDF fill lookup), `affine_from_geotransform` (GDAL `GeoTransform`)

::: geotoolz.io
