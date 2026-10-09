# API reference

## `geocloud.store`

The process-wide client pool and test mounts.

::: geocloud.store.SUPPORTED_SCHEMES
::: geocloud.store.get_obstore
::: geocloud.store.object_key
::: geocloud.store.get_range_bytes
::: geocloud.store.clear_obstore_pool
::: geocloud.store.set_obstore_pool_maxsize
::: geocloud.store.mount
::: geocloud.store.unmount

## `geocloud.files`

Whole-object verbs on the pool, cloud or local.

::: geocloud.files.ObjectInfo
::: geocloud.files.ls
::: geocloud.files.info
::: geocloud.files.exists
::: geocloud.files.read_bytes
::: geocloud.files.write_bytes
::: geocloud.files.open
::: geocloud.files.download
::: geocloud.files.upload
::: geocloud.files.copy
::: geocloud.files.sync
::: geocloud.files.rm
::: geocloud.files.sign

## `geocloud.credentials`

Credentials per store root, for the pool and GDAL.

::: geocloud.credentials.set_credentials
::: geocloud.credentials.remove_credentials
::: geocloud.credentials.credential_roots
::: geocloud.credentials.load_credentials
::: geocloud.credentials.credentials_path
::: geocloud.credentials.gdal_access
::: geocloud.credentials.GdalAccess
::: geocloud.credentials.redact

## `geocloud.cog`

COG reads (`[cog]` extra) and the validated COG writer.

::: geocloud.cog.CogSource
::: geocloud.cog.CogDomain
::: geocloud.cog.AsyncCogReader
::: geocloud.cog.read_from_window
::: geocloud.cog.read_from_bounds
::: geocloud.cog.read_from_polygon
::: geocloud.cog.read_from_center_coords
::: geocloud.cog.read_reproject
::: geocloud.cog.read_reproject_like
::: geocloud.cog.read_to_crs
::: geocloud.cog.read_from_tile
::: geocloud.cog.write_cog
