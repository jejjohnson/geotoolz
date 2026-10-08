"""`geocloud.credentials` — register credentials once per store root, then use URIs.

```python
from geocloud import credentials, files

credentials.set_credentials("s3://noaa-goes19", anonymous=True)
credentials.set_credentials("az://myaccount/raw", sas_token=sas)  # expiry checked
credentials.set_credentials("az://myaccount", use_azure_cli=True)  # other containers
credentials.load_credentials("team.toml")  # the default file loads by itself

files.download("az://myaccount/raw/scene.tif", "data/")  # no options to thread
path, env = credentials.gdal_access("az://myaccount/raw/scene.tif")  # for rasterio
```

Every package on the `geocloud.store` pool picks the credentials up.
Nothing is written to ``os.environ``; `redact` masks signatures and tokens
in text bound for logs.
"""

from __future__ import annotations

from geocloud._src.credentials import (
    GdalAccess,
    credential_roots,
    credentials_path,
    gdal_access,
    load_credentials,
    remove_credentials,
    set_credentials,
)
from geocloud._src.redact import redact


__all__ = [
    "GdalAccess",
    "credential_roots",
    "credentials_path",
    "gdal_access",
    "load_credentials",
    "redact",
    "remove_credentials",
    "set_credentials",
]
