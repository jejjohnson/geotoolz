"""Credential loading for data adapters.

We never write credentials — only read. Sources are checked in this
order:

1. Keyword arguments to the adapter constructor.
2. Environment variables (``COPERNICUSMARINE_SERVICE_USERNAME`` etc.).
3. Known on-disk config files (``~/.cmems``, ``~/.cdsapirc``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CMEMSCredentials:
    """Username / password pair for Copernicus Marine."""

    username: str
    password: str


@dataclass(frozen=True)
class CDSCredentials:
    """URL / key pair for the Climate Data Store."""

    url: str
    key: str


@dataclass(frozen=True)
class AEMETCredentials:
    """API key for AEMET OpenData."""

    api_key: str


def load_cmems(
    username: str | None = None,
    password: str | None = None,
    path: Path | None = None,
) -> CMEMSCredentials | None:
    """Load CMEMS credentials, or ``None`` if none could be found."""
    if username and password:
        return CMEMSCredentials(username=username, password=password)

    env_u = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    env_p = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if env_u and env_p:
        return CMEMSCredentials(username=env_u, password=env_p)

    cfg = path or Path.home() / ".cmems"
    if cfg.is_file():
        parsed = _parse_kv(cfg.read_text())
        u = parsed.get("username")
        p = parsed.get("password")
        if u and p:
            return CMEMSCredentials(username=u, password=p)

    return None


def load_cds(
    url: str | None = None,
    key: str | None = None,
    path: Path | None = None,
) -> CDSCredentials | None:
    """Load CDS credentials, or ``None`` if none could be found."""
    if url and key:
        return CDSCredentials(url=url, key=key)

    env_url = os.environ.get("CDSAPI_URL")
    env_key = os.environ.get("CDSAPI_KEY")
    if env_url and env_key:
        return CDSCredentials(url=env_url, key=env_key)

    cfg = path or Path.home() / ".cdsapirc"
    if cfg.is_file():
        parsed = _parse_kv(cfg.read_text())
        u = parsed.get("url")
        k = parsed.get("key")
        if u and k:
            return CDSCredentials(url=u, key=k)

    return None


def load_aemet(
    api_key: str | None = None,
    path: Path | None = None,
) -> AEMETCredentials | None:
    """Load AEMET credentials, or ``None`` if none could be found.

    Resolution order:

    1. Explicit ``api_key`` argument.
    2. ``AEMET_API_KEY`` environment variable.
    3. ``.env`` file in the current working directory (one
       ``AEMET_API_KEY=...`` line).
    4. ``path`` (or ``~/.aemet``) parsed as a ``key: value`` config.

    The ``.env`` step matches the most common local-dev workflow
    without introducing a ``python-dotenv`` dependency — we parse the
    single key we need and leave the file alone.
    """
    if api_key:
        return AEMETCredentials(api_key=api_key)

    env_key = os.environ.get("AEMET_API_KEY")
    if env_key:
        return AEMETCredentials(api_key=env_key)

    # Walk up from CWD looking for ``.env``; matches how
    # ``python-dotenv`` behaves so a notebook opened in ``docs/``
    # still picks up the project-root ``.env``.
    for candidate in (Path.cwd(), *Path.cwd().parents):
        dotenv = candidate / ".env"
        if dotenv.is_file():
            parsed = _parse_kv(dotenv.read_text())
            k = parsed.get("aemet_api_key")
            if k:
                return AEMETCredentials(api_key=k)
            break

    cfg = path or Path.home() / ".aemet"
    if cfg.is_file():
        parsed = _parse_kv(cfg.read_text())
        k = parsed.get("api_key") or parsed.get("aemet_api_key")
        if k:
            return AEMETCredentials(api_key=k)

    return None


def _parse_kv(text: str) -> dict[str, str]:
    """Parse ``key: value`` / ``key=value`` / ``key value`` config files."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (":", "=", " "):
            if sep in line:
                k, _, v = line.partition(sep)
                out[k.strip().lower()] = v.strip()
                break
    return out
