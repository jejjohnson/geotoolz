"""Mask signatures and tokens in text bound for logs and error messages."""

from __future__ import annotations

import re


_SECRET_PARAM = re.compile(
    r"(?i)([?&;](?:sig|signature|x-amz-signature|x-amz-credential|"
    r"x-amz-security-token|x-goog-signature|x-goog-credential|"
    r"access_token|token|api_key|apikey|key|code)=)[^&#\s'\"]*"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact(text: str) -> str:
    """``text`` with URL signatures, tokens and bearer credentials masked.

    Use it on any URI or message that may reach a log, an exception or a
    notebook output: SAS ``sig=``, S3 / GCS query signatures and
    credentials, ``token=`` / ``key=`` parameters and ``Bearer`` headers
    become ``REDACTED``. Everything else (paths, expiry times) is kept, so
    the redacted text still says which object and which window.

    Args:
        text: Any string.

    Returns:
        The string with secrets masked.

    Examples:
        >>> redact("https://a.blob.core.windows.net/c/x.tif?se=2026-11-01&sig=abc%3D")
        'https://a.blob.core.windows.net/c/x.tif?se=2026-11-01&sig=REDACTED'
        >>> redact("Authorization: Bearer eyJhbGciOi")
        'Authorization: Bearer REDACTED'
    """
    return _BEARER.sub(r"\1REDACTED", _SECRET_PARAM.sub(r"\1REDACTED", text))
