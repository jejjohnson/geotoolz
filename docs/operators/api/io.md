# IO

Reader and writer source/sink operators. For HDF / HDF-EOS / NetCDF-CF
reads and their extras, see [Read HDF and NetCDF files](../io.md).

- **Window / bounds readers:** `ReadWindow`, `ReadBounds`, `ReadCenterCoords`, `ReadPolygon`,
  `ReadTile`, `ReadToCRS`, `ReadReprojectLike`
- **Multi-format:** `ReadHDF` (HDF5/HDF4 dispatch), `ReadNetCDF` (CF mask/scale + group nav)
- **Cloud / catalog source:** `LoadFromEE`, `LoadFromSTAC`
- **Writers:** `WriteGeoTIFF`, `WriteCOG`, `WriteZarr`
- **Base classes:** `SourceOperator`, `SinkOperator`, `GeoToolzIOError`

`ReadHDF`, `ReadNetCDF` and `WriteCOG` are operators over geotoolz-cloud's
readers and writer ([`geocloud.hdf`](../../cloud/how-to/hdf.md),
[`geocloud.cog.write_cog`](../../cloud/how-to/write-cogs.md)), so they read
and write local paths and bucket URIs alike.

::: geotoolz.io
