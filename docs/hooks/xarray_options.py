"""Render the xarray lane's API pages with the mkdocstrings options xrtoolz used.

The xrops and xrproducts docstrings document constructor arguments on the
class and leave many return types to the prose, so the geotoolz defaults
would misrender them and fail ``mkdocs build --strict``. This hook gives
every ``:::`` directive under ``docs/xarray/`` xrtoolz's options, keeping any
option the page sets itself; the geotoolz pages are unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any


OPTIONS: dict[str, Any] = {
    "docstring_section_style": "table",
    "docstring_options": {"warn_unknown_params": False, "warn_missing_types": False},
    "merge_init_into_class": True,
    "show_root_full_path": False,
    "show_symbol_type_heading": True,
    "show_symbol_type_toc": True,
    "separate_signature": True,
    "show_signature_annotations": True,
    "signature_crossrefs": True,
    "members_order": "source",
    "filters": ["!^_"],
    "heading_level": 3,
}
DIRECTIVE = re.compile(r"^(?P<indent>\s*):::\s+\S+\s*$")
KEY = re.compile(r"^\s*(?P<key>\w+)\s*:")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def on_page_markdown(markdown: str, page: Any, **_: Any) -> str:
    if not page.file.src_uri.startswith("xarray/"):
        return markdown
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        match = DIRECTIVE.match(lines[i])
        i += 1
        if match is None:
            continue
        base = match["indent"] + "    "
        if (
            i < len(lines)
            and lines[i].strip() == "options:"
            and _indent(lines[i]) >= len(base)
        ):
            # keep the page's own options and add the missing ones after them
            options_indent = _indent(lines[i])
            out.append(lines[i])
            i += 1
            child = " " * (options_indent + 2)
            present: set[str] = set()
            while i < len(lines) and (
                not lines[i].strip() or _indent(lines[i]) > options_indent
            ):
                if lines[i].strip() and _indent(lines[i]) == len(child):
                    key = KEY.match(lines[i])
                    if key:
                        present.add(key["key"])
                out.append(lines[i])
                i += 1
            # trailing blank lines belong after the block
            blanks = 0
            while out and not out[-1].strip():
                out.pop()
                blanks += 1
            out.extend(
                f"{child}{k}: {json.dumps(v)}"
                for k, v in OPTIONS.items()
                if k not in present
            )
            out.extend([""] * blanks)
        else:
            out.append(f"{base}options:")
            out.extend(f"{base}  {k}: {json.dumps(v)}" for k, v in OPTIONS.items())
    return "\n".join(out)
