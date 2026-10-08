"""Credential storage shared by provider clients.

Every authenticated provider keeps its credentials in one owner-only JSON
file, ``~/.geoproducts/auth_<provider>.json``, with environment variables
taking priority. This module holds the parts that do not depend on the
provider: the path convention, owner-only writes, reading, and JWT expiry.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from geoproducts._src.files import write_private_json


__all__ = ["auth_path", "jwt_expiry", "read_json_config", "write_private_json"]


def auth_path(provider: str) -> Path:
    """The canonical credentials file of ``provider``.

    Examples:
        >>> auth_path("carbonmapper").name
        'auth_carbonmapper.json'
    """
    return Path.home() / ".geoproducts" / f"auth_{provider}.json"


def read_json_config(path: Path | str) -> dict[str, Any]:
    """A JSON object from ``path`` (``~`` expanded).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file is not a JSON object.
    """
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} holds {type(data).__name__}, not a JSON object.")
    return data


def jwt_expiry(token: str) -> float | None:
    """The ``exp`` claim (Unix seconds) of a JWT, or ``None``.

    Decodes the payload only — no signature check; the issuing API
    validates the token itself. ``None`` when ``token`` is not a JWT with a
    numeric ``exp``.

    Examples:
        >>> jwt_expiry("not-a-jwt") is None
        True
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return float(exp) if isinstance(exp, int | float) else None
