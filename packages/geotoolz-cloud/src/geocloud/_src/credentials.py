"""Per-store credentials for the pool, a GDAL bridge, and secret redaction.

Credentials are registered once per **store root** — an S3 / GCS bucket,
an Azure account or container, an HTTP host — and `get_obstore` merges
them under any ``storage_options`` a caller passes. Code that moves or
reads data then only ever handles URIs.

- `set_credentials` / `remove_credentials` / `credential_roots` — the
  registry. ``anonymous`` and ``sas_token`` are spelled out (a SAS is
  checked for expiry and scope when registered); every other keyword is
  an obstore store option
  (``aws_access_key_id``, ``account_key``, ``use_azure_cli``,
  ``credential_provider``, ``client_options``, …).
- `load_credentials` — register every root in a TOML file; ``${VAR}`` in a value
  reads the environment. The file at ``$GEOCLOUD_CREDENTIALS`` (default
  ``~/.config/geocloud/credentials.toml``) is loaded on first use.
- `gdal_access` — the GDAL path and config options for a URI, from the
  same registry, for rasterio / georeader readers.
- `redact` — mask signatures and tokens in text bound for logs or errors.

Nothing here writes ``os.environ``: credentials live in the registry and
reach obstore as store options and GDAL as explicit config options.
Azure managed identity, workload identity and S3 / GCS instance
credentials need no registration — obstore's default chains find them.
"""

from __future__ import annotations

import os
import re
import stat
import threading
import tomllib
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qsl, quote, urlsplit

from geocloud._src.redact import redact
from geocloud._src.store import (
    _AZ_SCHEMES,
    _AZURE_HOST_SUFFIXES,
    _HTTP_SCHEMES,
    _build_store,
    _locate,
)


__all__ = [
    "GdalAccess",
    "credential_roots",
    "credentials_path",
    "gdal_access",
    "load_credentials",
    "remove_credentials",
    "set_credentials",
]

#: ``(backend, bucket / account / host, container or None)``.
_Scope = tuple[str, str, str | None]

_REGISTRY: dict[_Scope, dict[str, Any]] = {}
_LOCK = threading.RLock()
_AUTOLOADED = False

# Spellings obstore accepts for an Azure SAS; all are routed through the
# SAS checks below.
_SAS_KEYS = ("sas_token", "sas_key", "azure_storage_sas_key", "azure_storage_sas_token")


# --- scopes ------------------------------------------------------------------


def _scope(uri: str) -> _Scope:
    """The store root ``uri`` names; rejects a URI that names an object."""
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    bare = not parsed.path.strip("/")
    if bare and scheme in _AZ_SCHEMES and parsed.netloc:
        return ("azure", parsed.netloc, None)  # az://account — every container
    if bare and scheme in _HTTP_SCHEMES and host.endswith(_AZURE_HOST_SUFFIXES):
        return ("azure", host.split(".", 1)[0], None)
    if scheme == "hf":
        raise ValueError(
            "credentials: hf:// URIs use the Hugging Face token ($HF_TOKEN or "
            "`huggingface-cli login`); nothing to register."
        )
    loc = _locate(uri)
    if loc.key:
        raise ValueError(
            f"credentials: register a store root (`s3://bucket`, "
            f"`az://account[/container]`, `https://host`), not the object "
            f"{redact(uri)!r}."
        )
    if loc.backend == "http":
        return ("http", loc.bucket, None)
    return (loc.backend, loc.bucket, loc.scope)


def _scope_uri(scope: _Scope) -> str:
    backend, bucket, container = scope
    if backend == "azure":
        return f"az://{bucket}" + (f"/{container}" if container else "")
    return {"s3": "s3", "gcs": "gs", "http": "https"}[backend] + f"://{bucket}"


# --- SAS checks ------------------------------------------------------------------


def _sas_params(token: str) -> dict[str, str]:
    return {
        k.lower(): v for k, v in parse_qsl(token.lstrip("?"), keep_blank_values=True)
    }


def _sas_is_service_scoped(token: str) -> bool:
    """A service SAS (container or blob, ``sr=``) rather than an account SAS."""
    params = _sas_params(token)
    return "sr" in params and "srt" not in params


def _check_sas(token: str, scope: _Scope) -> str:
    """Validate a SAS for ``scope``; return it without a leading ``?``."""
    token = token.strip().lstrip("?")
    params = _sas_params(token)
    where = _scope_uri(scope)
    if "sig" not in params:
        raise ValueError(f"credentials: the SAS token for {where} has no `sig=`.")
    if expiry := params.get("se"):
        try:
            expires = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"credentials: the SAS token for {where} has an unreadable "
                f"expiry `se={expiry}`."
            ) from exc
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires <= datetime.now(UTC):
            raise ValueError(
                f"credentials: the SAS token for {where} expired at {expiry}."
            )
    if scope[2] is None and _sas_is_service_scoped(token):
        raise ValueError(
            f"credentials: the SAS token for {where} is container-scoped "
            f"(`sr={params['sr']}`); register it at `az://{scope[1]}/<container>`."
        )
    return token


# --- registry --------------------------------------------------------------------


def set_credentials(
    uri: str,
    *,
    anonymous: bool = False,
    sas_token: str | None = None,
    **storage_options: Any,
) -> None:
    """Use these credentials for every object under the store root ``uri``.

    `get_obstore` (and so `geocloud.files`, `geocloud.cog` and every
    package on the pool) merges them under the ``storage_options`` a call
    passes; an Azure container's registration wins over its account's.
    The options are checked by building a store now, so an unknown key or
    a missing key file fails here rather than at first read. Registering
    a root again replaces its entry.

    Args:
        uri: The store root: ``s3://bucket``, ``gs://bucket``,
            ``az://account`` (all its containers) or
            ``az://account/container``, ``https://host``.
        anonymous: Send unsigned requests (public buckets and containers);
            no credential lookup, no instance-metadata probe.
        sas_token: An Azure SAS token. Rejected when it has expired, or
            when a container-scoped SAS is registered for a whole account.
        **storage_options: Any other obstore store option for the
            backend (``aws_access_key_id`` / ``aws_secret_access_key`` /
            ``region``, ``account_key``, ``use_azure_cli``,
            ``service_account``, ``credential_provider``,
            ``client_options``, …).

    Raises:
        ValueError: ``uri`` names an object or an ``hf://`` repo, an
            option does not apply to the backend, or the SAS is expired /
            mis-scoped.

    Examples:
        >>> set_credentials("s3://noaa-goes19", anonymous=True, region="us-east-1")
        >>> credential_roots()
        ['s3://noaa-goes19']
        >>> remove_credentials("s3://noaa-goes19")
    """
    scope = _scope(uri)
    options = dict(storage_options)
    for name in _SAS_KEYS:
        if name in options:
            if sas_token is not None:
                raise ValueError(f"credentials: pass the SAS once, not also as {name}.")
            sas_token = options.pop(name)
    if sas_token is not None:
        if scope[0] != "azure":
            raise ValueError("credentials: `sas_token` applies to Azure roots only.")
        options["sas_key"] = _check_sas(sas_token, scope)
    if anonymous:
        if scope[0] == "http":
            raise ValueError(
                "credentials: plain HTTP stores are already anonymous unless "
                "given headers; drop `anonymous`."
            )
        options["skip_signature"] = True
    if scope[0] != "http":
        probe = _scope_uri(
            scope if scope[2] or scope[0] != "azure" else (*scope[:2], "probe")
        )
        try:
            _build_store(probe, options)
        except Exception as exc:
            raise ValueError(
                f"credentials: invalid options for {_scope_uri(scope)}: "
                f"{redact(str(exc).split(chr(10), 1)[0])}"
            ) from None
    with _LOCK:
        _REGISTRY[scope] = options


def remove_credentials(uri: str) -> None:
    """Forget the credentials registered at the root ``uri`` (no-op if none)."""
    with _LOCK:
        _REGISTRY.pop(_scope(uri), None)


def credential_roots() -> list[str]:
    """The store roots that have credentials registered (never the secrets).

    Examples:
        >>> isinstance(credential_roots(), list)
        True
    """
    with _LOCK:
        _autoload()
        return sorted(_scope_uri(scope) for scope in _REGISTRY)


def _clear() -> None:
    """Empty the registry and allow the default file to load again (tests)."""
    global _AUTOLOADED
    with _LOCK:
        _REGISTRY.clear()
        _AUTOLOADED = False


def registered_options(uri: str) -> dict[str, Any]:
    """The registered options that apply to ``uri`` (a copy; ``{}`` if none)."""
    if urlsplit(uri).scheme.lower() == "hf":
        return {}
    loc = _locate(uri)
    container = loc.scope if loc.backend == "azure" else None
    with _LOCK:
        _autoload()
        for key in (
            (loc.backend, loc.bucket, container),
            (loc.backend, loc.bucket, None),
        ):
            if key in _REGISTRY:
                return dict(_REGISTRY[key])
    return {}


# --- files -----------------------------------------------------------------------

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, where: str) -> Any:
    """Replace ``${VAR}`` in strings (recursively); an unset VAR is an error."""
    if isinstance(value, str):

        def env(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"credentials: {where} reads ${{{name}}}, which is unset."
                )
            return os.environ[name]

        return _ENV_REF.sub(env, value)
    if isinstance(value, Mapping):
        return {k: _expand(v, where) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, where) for v in value]
    return value


def credentials_path() -> Path | None:
    """The default credentials file, or ``None`` when turned off.

    ``$GEOCLOUD_CREDENTIALS`` when set (empty turns the file off), else
    ``$XDG_CONFIG_HOME/geocloud/credentials.toml`` (``~/.config/…``).
    """
    explicit = os.environ.get("GEOCLOUD_CREDENTIALS")
    if explicit is not None:
        return Path(explicit).expanduser() if explicit else None
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config).expanduser() / "geocloud" / "credentials.toml"


def load_credentials(path: str | os.PathLike[str] | None = None) -> list[str]:
    """Register every store root in a TOML credentials file.

    Each table is a store root; its keys are `set_credentials`'s keywords.
    ``${VAR}`` inside a string value is read from the environment, so the
    file can hold references rather than secrets::

        ["s3://noaa-goes19"]
        anonymous = true
        region = "us-east-1"

        ["az://myaccount/raw"]
        sas_token = "${RAW_SAS}"

        ["s3://private-bucket"]
        aws_access_key_id = "${AWS_KEY}"
        aws_secret_access_key = "${AWS_SECRET}"
        region = "eu-west-1"

    A file other users can read triggers a warning.

    Args:
        path: The file; ``None`` uses `credentials_path` (and returns ``[]``
            when it does not exist).

    Returns:
        The roots registered.

    Raises:
        FileNotFoundError: An explicit ``path`` does not exist.
        ValueError: A table is not a root, a ``${VAR}`` is unset, or
            `set_credentials` rejects an entry (the message names the root).
    """
    target = Path(path).expanduser() if path is not None else credentials_path()
    if target is None or (path is None and not target.is_file()):
        return []
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    if os.name == "posix" and target.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        warnings.warn(
            f"{target} is readable by other users; `chmod 600` it.",
            stacklevel=2,
        )
    roots: list[str] = []
    for root, entry in data.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"credentials: {target}: `{root}` must be a table.")
        set_credentials(root, **_expand(dict(entry), f"{target} [{root}]"))
        roots.append(root)
    return roots


def _autoload() -> None:
    """Load the default credentials file once per process (caller holds the lock)."""
    global _AUTOLOADED
    if _AUTOLOADED:
        return
    _AUTOLOADED = True
    load_credentials()


# --- GDAL --------------------------------------------------------------------------


class GdalAccess(NamedTuple):
    """How GDAL (rasterio, georeader) reads a URI with the registered credentials.

    Attributes:
        path: The GDAL path (``/vsis3/…``, ``/vsigs/…``, ``/vsiaz/…``,
            ``/vsicurl/…`` or a local path).
        options: GDAL config options for ``rasterio.Env(**options)`` or
            georeader's ``rio_env_options``.
    """

    path: str
    options: dict[str, str]


def _true(value: Any) -> bool:
    return str(value).lower() in ("true", "1", "yes")


def gdal_access(uri: str) -> GdalAccess:
    """GDAL path and config options to read ``uri`` with its registered credentials.

    The options come from the same registry as `get_obstore`, so a root
    registered once reads through both obstore and GDAL. Nothing is set in
    ``os.environ``; pass the options to ``rasterio.Env`` or to georeader::

        path, env = gdal_access("az://account/raw/scene.tif")
        reader = RasterioReader(path, rio_env_options=env)
        with rasterio.Env(**env), rasterio.open(path) as src: ...

    A container-scoped Azure SAS is put in an HTTPS URL (``/vsicurl/``),
    because GDAL's ``/vsiaz/`` expects an account-level grant; an
    account SAS or key goes to ``/vsiaz/`` as config options. Credentials
    neither side can express (a credential-provider object, an inline
    service-account key) raise: pre-sign the object with
    `geocloud.files.sign` and read ``/vsicurl/<url>`` instead.

    Args:
        uri: A ``s3://``, ``gs://``, Azure, ``http(s)://`` URI or a local
            path.

    Returns:
        A `GdalAccess` ``(path, options)``.

    Raises:
        ValueError: ``hf://`` URIs, mounted stores, or credentials GDAL
            cannot take as config options.

    Examples:
        >>> set_credentials("s3://open-data", anonymous=True, region="us-west-2")
        >>> path, options = gdal_access("s3://open-data/a/b.tif")
        >>> path
        '/vsis3/open-data/a/b.tif'
        >>> options
        {'AWS_NO_SIGN_REQUEST': 'YES', 'AWS_REGION': 'us-west-2'}
        >>> remove_credentials("s3://open-data")
    """
    scheme = urlsplit(uri).scheme.lower()
    if "://" not in uri or scheme == "file":
        return GdalAccess(urlsplit(uri).path if scheme == "file" else uri, {})
    if scheme == "hf":
        raise ValueError(
            "gdal_access: hf:// is not a GDAL path; read it with geocloud.cog."
        )
    loc = _locate(uri)
    if loc.backend == "http":
        return GdalAccess(f"/vsicurl/{uri}", {})

    from geocloud._src.store import get_obstore

    store = get_obstore(uri)
    config: dict[str, Any] = dict(getattr(store, "config", None) or {})
    if not config:
        raise ValueError(
            f"gdal_access: {redact(uri)!r} is served by a mounted store; GDAL "
            "cannot read it."
        )
    if getattr(store, "credential_provider", None) is not None:
        raise ValueError(
            "gdal_access: a credential provider cannot be handed to GDAL; "
            "pre-sign with geocloud.files.sign and read /vsicurl/<url>."
        )
    options: dict[str, str] = {}
    if loc.backend == "s3":
        for key, name in (
            ("access_key_id", "AWS_ACCESS_KEY_ID"),
            ("secret_access_key", "AWS_SECRET_ACCESS_KEY"),
            ("session_token", "AWS_SESSION_TOKEN"),
            ("region", "AWS_REGION"),
        ):
            if key in config:
                options[name] = str(config[key])
        if _true(config.get("skip_signature")):
            options["AWS_NO_SIGN_REQUEST"] = "YES"
        if _true(config.get("request_payer")):
            options["AWS_REQUEST_PAYER"] = "requester"
        if endpoint := config.get("endpoint"):
            parts = urlsplit(str(endpoint))
            options["AWS_S3_ENDPOINT"] = parts.netloc or str(endpoint)
            options["AWS_HTTPS"] = "NO" if parts.scheme == "http" else "YES"
            options["AWS_VIRTUAL_HOSTING"] = (
                "TRUE" if _true(config.get("virtual_hosted_style_request")) else "FALSE"
            )
        return GdalAccess(
            f"/vsis3/{loc.bucket}/{loc.key}", dict(sorted(options.items()))
        )
    if loc.backend == "gcs":
        if "service_account_key" in config:
            raise ValueError(
                "gdal_access: an inline service-account key cannot be handed to "
                "GDAL; register `service_account` (a key file path) instead."
            )
        if "service_account" in config:
            options["GOOGLE_APPLICATION_CREDENTIALS"] = str(config["service_account"])
        if _true(config.get("skip_signature")):
            options["GS_NO_SIGN_REQUEST"] = "YES"
        return GdalAccess(f"/vsigs/{loc.bucket}/{loc.key}", options)
    # Azure
    account, container = loc.bucket, loc.scope
    sas = config.get("sas_key")
    if sas and _sas_is_service_scoped(str(sas)):
        url = f"https://{account}.blob.core.windows.net/{container}/{quote(loc.key)}"
        return GdalAccess(f"/vsicurl/{url}?{sas}", {})
    options["AZURE_STORAGE_ACCOUNT"] = account
    if sas:
        options["AZURE_STORAGE_SAS_TOKEN"] = str(sas)
    if "account_key" in config:
        options["AZURE_STORAGE_ACCESS_KEY"] = str(config["account_key"])
    if _true(config.get("skip_signature")):
        options["AZURE_NO_SIGN_REQUEST"] = "YES"
    return GdalAccess(f"/vsiaz/{container}/{loc.key}", dict(sorted(options.items())))
