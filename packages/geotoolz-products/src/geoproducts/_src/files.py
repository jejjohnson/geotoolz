"""Atomic file writes shared by the product readers.

Downloads, caches and credential files all need the same guarantee: the
final name only ever holds a complete file. :func:`atomic_path` yields a
temporary sibling to write and renames it into place on success (removing
it on failure); ``private=True`` creates it owner-only (``0600``) from the
first byte, for secrets.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


__all__ = ["atomic_path", "atomic_write_text", "write_private_json"]


@contextmanager
def atomic_path(dest: Path | str, *, private: bool = False) -> Iterator[Path]:
    """Yield a temporary sibling of ``dest`` that replaces it on success.

    Public files use ``<name>.part``; private ones a unique hidden
    ``mkstemp`` file, created ``0600`` so the content is never readable
    by other users, not even briefly. Either way the temporary file is
    removed if the block raises, and ``dest``'s directory is created.

    Args:
        dest: Final path.
        private: Create the file owner-only (credentials, tokens).

    Yields:
        The temporary path to write.

    Examples:
        >>> import tempfile, pathlib
        >>> target = pathlib.Path(tempfile.mkdtemp()) / "out.txt"
        >>> with atomic_path(target) as tmp:
        ...     _ = tmp.write_text("done")
        >>> target.read_text()
        'done'
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if private:
        fd, name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.")
        os.close(fd)
        tmp = Path(name)
    else:
        tmp = dest.with_name(dest.name + ".part")
    try:
        yield tmp
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(dest: Path | str, text: str, *, private: bool = False) -> Path:
    """Write ``text`` to ``dest`` atomically (see :func:`atomic_path`)."""
    dest = Path(dest)
    with atomic_path(dest, private=private) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return dest


def write_private_json(dest: Path | str, data: Any) -> Path:
    """Write ``data`` as indented JSON to an owner-only (``0600``) file."""
    return atomic_write_text(dest, json.dumps(data, indent=2), private=True)
