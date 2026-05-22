# IO

Reader and writer source/sink operators. See the concept page
[Multi-format readers](../io.md) for the HDF / HDF-EOS / NetCDF-CF stack and the install extras
(`geotoolz[hdf5]`, `geotoolz[hdf4]`, `geotoolz[netcdf]`).

- **Window / bounds readers:** `ReadWindow`, `ReadBounds`, `ReadCenterCoords`, `ReadPolygon`,
  `ReadTile`, `ReadToCRS`, `ReadReprojectLike`
- **Multi-format:** `ReadHDF` (HDF5/HDF4 dispatch), `ReadNetCDF` (CF mask/scale + group nav)
- **Cloud / catalog source:** `LoadFromEE`, `LoadFromSTAC`
- **Writers:** `WriteGeoTIFF`, `WriteCOG`, `WriteZarr`
- **Base classes:** `SourceOperator`, `SinkOperator`, `GeoToolzIOError`

::: geotoolz.io
