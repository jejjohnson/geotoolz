"""Climate Data Store (CDS) adapter built on ``cdsapi``.

The underlying client is imported lazily so ``xr_toolz.data`` can be
imported without the optional ``cdsapi`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from xr_toolz.data._src.base import DatasetInfo, DataSource
from xr_toolz.data._src.cds.catalog import CDS_DATASETS
from xr_toolz.data._src.cds.profiles import resolve_profile
from xr_toolz.data._src.credentials import CDSCredentials, load_cds
from xr_toolz.types import (
    BBox,
    DepthRange,
    PressureLevels,
    TimeRange,
    Variable,
)


class CDSSource(DataSource):
    """Adapter around the ``cdsapi`` Python client.

    Args:
        credentials: Explicit :class:`CDSCredentials`. When ``None``,
            credentials are resolved from env vars / ``~/.cdsapirc``.
        client: Optional pre-built ``cdsapi.Client`` (or test double).
        format: CDS output format. When ``None`` (default) the family
            profile picks — ``"netcdf"`` for reanalyses, ``"zip"`` for
            in-situ. Override with ``"grib"`` / ``"netcdf"`` / ``"zip"``
            to force a format for every request from this source.
        product_type: Default ``product_type`` form entry when the
            dataset's profile admits it (reanalysis datasets; ignored
            for in-situ).
    """

    source_id = "cds"

    def __init__(
        self,
        credentials: CDSCredentials | None = None,
        client: Any | None = None,
        format: str | None = None,
        product_type: str | None = "reanalysis",
    ) -> None:
        self.credentials = credentials or load_cds()
        self._client = client
        self.format = format
        self.product_type = product_type

    # ---- client handling --------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import cdsapi  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "CDSSource requires the 'cdsapi' package. "
                "Install with: pip install xr_toolz[data]"
            ) from exc
        kwargs: dict[str, str] = {}
        if self.credentials is not None:
            kwargs["url"] = self.credentials.url
            kwargs["key"] = self.credentials.key
        self._client = cdsapi.Client(**kwargs)
        return self._client

    # ---- DataSource API ---------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        return list(CDS_DATASETS.values())

    def describe(self, dataset_id: str) -> DatasetInfo:
        if dataset_id in CDS_DATASETS:
            return CDS_DATASETS[dataset_id]
        return DatasetInfo(
            dataset_id=dataset_id,
            source=self.source_id,
            title=dataset_id,
        )

    def download(
        self,
        dataset_id: str,
        output: Path,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        **extras: Any,
    ) -> Path:
        """Retrieve a dataset to ``output`` via ``cdsapi.Client.retrieve``."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        form = self._build_form(
            dataset_id=dataset_id,
            variables=variables,
            bbox=bbox,
            time=time,
            levels=levels,
            extras=extras,
        )
        self._get_client().retrieve(dataset_id, form, str(output))
        return output

    def open(
        self,
        dataset_id: str,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        **extras: Any,
    ) -> xr.Dataset:
        """Download then open the resulting file with ``xarray.open_dataset``.

        CDS has no native lazy access path, so this always materialises
        a local file (under the request cache) before handing back an
        ``xr.Dataset``. Both the cache file suffix and the xarray engine
        follow :attr:`format`, so reconfiguring to ``"grib"`` remains
        consistent end-to-end.
        """
        from xr_toolz.data._src.cache import cache_path

        request = {
            "dataset_id": dataset_id,
            "variables": [v if isinstance(v, str) else v.name for v in variables or []],
            "bbox": bbox.__dict__ if bbox else None,
            "time": {
                "start": time.start.isoformat(),
                "end": time.end.isoformat(),
                "freq": time.freq,
            }
            if time is not None
            else None,
            "levels": levels.levels if levels else None,
            "extras": extras,
        }
        # Same format-resolution order as ``_build_form``. Respect the
        # profile's ``format_key`` (``format`` vs ``data_format``) and
        # treat ``None`` as "not provided" so a stale kwarg can't blank
        # the resolved format out.
        profile = resolve_profile(dataset_id)
        candidate = extras.get(profile.format_key) or extras.get("format")
        resolved_format = (
            candidate
            if isinstance(candidate, str) and candidate
            else (self.format or profile.format_default)
        )
        # Fail fast before the (potentially long, CDS-queued) download
        # when the output format isn't xarray-readable — use
        # ``CDSInsituArchive`` or ``download()`` for those.
        if resolved_format.lower() in {"zip", "csv"}:
            raise ValueError(
                f"CDS dataset {dataset_id!r} returns a {resolved_format} "
                "bundle, not an xarray-readable file. Use CDSInsituArchive "
                "or download() directly and parse the result yourself."
            )
        suffix = _suffix_for_format(resolved_format)
        path = cache_path(self.source_id, dataset_id, request, suffix=suffix)
        if not path.exists():
            self.download(
                dataset_id,
                path,
                variables=variables,
                bbox=bbox,
                time=time,
                depth=depth,
                levels=levels,
                **extras,
            )
        engine = _engine_for_format(resolved_format)
        return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)

    # ---- payload construction --------------------------------------------

    def _build_form(
        self,
        *,
        dataset_id: str,
        variables: list[str | Variable] | None,
        bbox: BBox | None,
        time: TimeRange | None,
        levels: PressureLevels | None,
        extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Shape the ``cdsapi`` form for ``dataset_id`` per its profile.

        Priority for ``format`` / ``product_type`` / fixed defaults:
        caller-supplied ``extras`` > source-level override > profile default.
        The profile's ``format_key`` ("format" for reanalysis,
        "data_format" for in-situ) determines which field name the
        output format goes under.
        """
        profile = resolve_profile(dataset_id)
        # Work on a copy — tests assert the outer extras dict isn't mutated.
        extras = dict(extras)
        form: dict[str, Any] = {}

        # Start with profile-fixed baseline (e.g. {"version": "2_0_0"}).
        # Extras override on the same key.
        for fk, fv in profile.fixed.items():
            form[fk] = extras.pop(fk, fv)

        # Output format: caller > source > profile default. The caller
        # may pass the override under either the profile's
        # ``format_key`` ("data_format" for in-situ) or the generic
        # alias ``"format"`` — both are recognised. ``None`` is treated
        # as "not provided" so a stale ``format=None`` kwarg doesn't
        # blank out the form key.
        fmt_key = profile.format_key
        caller_fmt: Any = None
        for alias in (fmt_key, "format"):
            if alias in extras:
                caller_fmt = extras.pop(alias)
                if caller_fmt is not None:
                    break
        if isinstance(caller_fmt, str) and caller_fmt:
            form[fmt_key] = caller_fmt
        elif self.format is not None:
            form[fmt_key] = self.format
        else:
            form[fmt_key] = profile.format_default

        if profile.includes_product_type:
            pt = extras.pop("product_type", self.product_type)
            if pt is not None:
                form["product_type"] = pt
        elif "product_type" in extras:
            # The dataset family has no ``product_type`` form key —
            # forwarding a caller's value would make CDS reject the
            # request. Surface the mismatch clearly.
            raise ValueError(
                f"CDS dataset {dataset_id!r} (profile={profile.family!r}) "
                "does not accept the 'product_type' form key; remove the "
                "product_type argument for this dataset."
            )

        if profile.required_extras:
            missing = [k for k in profile.required_extras if k not in extras]
            if missing:
                raise ValueError(
                    f"CDS dataset {dataset_id!r} (profile={profile.family!r}) "
                    f"requires form key(s) {missing!r}; pass them as keyword "
                    "arguments to download()/open()."
                )

        if variables:
            form["variable"] = self._encode_variables(variables)
        if bbox is not None and profile.uses_area:
            form["area"] = bbox.as_cds_area()
        if time is not None:
            time_form: dict[str, Any] = dict(time.as_cds_form())
            if not profile.year_is_array:
                # In-situ: ``year`` is a single string. We require a
                # single-year window; callers needing multi-year pull
                # should loop (the archive does this by year-chunking).
                years = time_form.get("year", [])
                if len(years) > 1:
                    raise ValueError(
                        f"CDS dataset {dataset_id!r} (profile="
                        f"{profile.family!r}) accepts one year per request; "
                        f"got {years!r}."
                    )
                if years:
                    time_form["year"] = years[0]
            form.update(time_form)
        if levels is not None and profile.uses_pressure_level:
            form["pressure_level"] = levels.as_cds_form()
        form.update(extras)
        return form


# ---- format <-> filesystem / xarray glue ------------------------------------


_FORMAT_SUFFIX: dict[str, str] = {
    "netcdf": ".nc",
    "netcdf4": ".nc",
    "nc": ".nc",
    "grib": ".grib",
    "grib2": ".grib",
    # CDS in-situ downloads: ``data_format=csv`` usually returns a zip
    # of CSVs, not a bare CSV, but we treat both the same on-disk.
    "zip": ".zip",
    "csv": ".zip",
}

_FORMAT_ENGINE: dict[str, str | None] = {
    "netcdf": None,  # xarray picks the best netcdf engine
    "netcdf4": None,
    "nc": None,
    "grib": "cfgrib",
    "grib2": "cfgrib",
    # No xarray engine for raw zip / csv bundles — CDSInsituArchive
    # unpacks and parses them.
    "zip": None,
    "csv": None,
}


def _suffix_for_format(fmt: str) -> str:
    """File suffix to use when caching a CDS download of format ``fmt``."""
    try:
        return _FORMAT_SUFFIX[fmt.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CDS format {fmt!r}; known: {sorted(_FORMAT_SUFFIX)}"
        ) from exc


def _engine_for_format(fmt: str) -> str | None:
    """xarray engine to use when opening a cached file of format ``fmt``."""
    return _FORMAT_ENGINE.get(fmt.lower())
