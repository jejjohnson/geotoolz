# Register credentials

Register credentials once per store root, then pass only URIs. Every
call on the pool picks them up, and `gdal_access` hands GDAL the same
grant. Which entry wins when several apply is in
[Concepts](../concepts.md#credential-precedence).

## Register a root

```python
import os

from geocloud import credentials, files

credentials.set_credentials("s3://noaa-goes19", anonymous=True, region="us-east-1")
credentials.set_credentials("az://myaccount/raw", sas_token=os.environ["RAW_SAS"])  # one container
credentials.set_credentials("az://myaccount", use_azure_cli=True)                   # its other containers
credentials.set_credentials(
    "s3://private",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region="eu-west-1",
)
credentials.set_credentials(
    "https://data.example.com",
    client_options={"default_headers": {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}},
)

files.copy("az://myaccount/raw/scene.tif", "s3://private/scenes/")  # each end its own grant
roots: list[str] = credentials.credential_roots()  # ['az://myaccount', 'az://myaccount/raw', …]
```

The secrets come from the environment; set `RAW_SAS`, the AWS keys and
`API_TOKEN` before running it.

- **`anonymous=True`** sends unsigned requests, with no credential lookup
  and no instance-metadata probe.
- **`sas_token=`** is checked at registration. It is rejected when
  expired, missing `sig=`, or container-scoped but registered for a whole
  account.
- **Anything else** is an obstore store option: keys, `region`, `endpoint`,
  `use_azure_cli`, `service_account`, `credential_provider`,
  `client_options`. A store is built to validate them, so a typo fails at
  `set_credentials`.

Azure managed identity and S3 / GCS instance credentials need no
registration: obstore's default chains find them.

## Load a credentials file

`load_credentials(path)` registers every table of a TOML file. A value
can reference the environment with `${VAR}` instead of holding the
secret.

```toml
["s3://noaa-goes19"]
anonymous = true
region = "us-east-1"

["az://myaccount/raw"]
sas_token = "${RAW_SAS}"

["s3://private"]
aws_access_key_id = "${AWS_KEY}"
aws_secret_access_key = "${AWS_SECRET}"
region = "eu-west-1"
```

```python
import os
import tempfile
from pathlib import Path

from geocloud import credentials

path: Path = Path(tempfile.mkdtemp()) / "credentials.toml"
path.write_text('["s3://sentinel-cogs"]\nanonymous = true\nregion = "us-west-2"\n')
path.chmod(0o600)                                    # a group- or world-readable file warns
loaded: list[str] = credentials.load_credentials(path)   # ['s3://sentinel-cogs']
default: Path | None = credentials.credentials_path()    # ~/.config/geocloud/credentials.toml
```

The default file loads by itself on first use. It is
`$GEOCLOUD_CREDENTIALS` when set, else
`~/.config/geocloud/credentials.toml`; set `GEOCLOUD_CREDENTIALS=` (empty)
to turn it off.

## Read through GDAL, rasterio or georeader

`gdal_access(uri)` turns the registry into a GDAL path plus config
options, so one registration serves obstore and GDAL alike.

```python
import rasterio
from georeader.rasterio_reader import RasterioReader

from geocloud import credentials
from geocloud.credentials import GdalAccess

credentials.set_credentials("s3://sentinel-cogs", anonymous=True, region="us-west-2")

access: GdalAccess = credentials.gdal_access(
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/10/S/DG/2024/7/S2A_10SDG_20240702_0_L2A/B04.tif"
)  # path='/vsis3/sentinel-cogs/…/B04.tif', options={'AWS_NO_SIGN_REQUEST': 'YES', 'AWS_REGION': 'us-west-2'}
reader: RasterioReader = RasterioReader(access.path, rio_env_options=access.options)  # (1, 10980, 10980) uint16
with rasterio.Env(**access.options), rasterio.open(access.path) as src:
    overviews: list[int] = src.overviews(1)          # [2, 4, 8, 16]
```

| Registered | GDAL path | Options |
| --- | --- | --- |
| S3 keys / anonymous / endpoint | `/vsis3/bucket/key` | `AWS_*` (`AWS_NO_SIGN_REQUEST`, `AWS_S3_ENDPOINT`, …) |
| GCS service-account file / anonymous | `/vsigs/bucket/key` | `GOOGLE_APPLICATION_CREDENTIALS`, `GS_NO_SIGN_REQUEST` |
| Azure account key / account SAS / anonymous | `/vsiaz/container/key` | `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_ACCESS_KEY` / `_SAS_TOKEN` |
| Azure **container** SAS | `/vsicurl/https://account.blob.core.windows.net/container/key?<sas>` | none |
| signed `http(s)://` URL | `/vsicurl/<url>` | none |

Credential-provider objects and inline service-account keys cannot be
passed to GDAL and raise. Pre-sign the object with
[`files.sign`](move-files.md#share-without-credentials) and read
`/vsicurl/<url>` instead.

## Keep secrets out of logs

`redact(text)` masks SAS signatures, S3 / GCS query signatures,
`token=` / `key=` parameters and `Bearer` headers. Every error geocloud
raises about a URI already goes through it.

```python
from geocloud.credentials import redact

line: str = redact("GET https://acct.blob.core.windows.net/c/x.tif?sv=2024&sig=abcd%3D")
# 'GET https://acct.blob.core.windows.net/c/x.tif?sv=2024&sig=REDACTED'
```
