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
        format: CDS output format, default ``"netcdf"``.
        product_type: Default ``product_type`` form entry when one
            isn't provided per-call.
    """

    source_id = "cds"

    def __init__(
        self,
        credentials: CDSCredentials | None = None,
        client: Any | None = None,
        format: str = "netcdf",
        product_type: str = "reanalysis",
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
        suffix = _suffix_for_format(self.format)
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
        engine = _engine_for_format(self.format)
        return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)

    # ---- payload construction --------------------------------------------

    def _build_form(
        self,
        variables: list[str | Variable] | None,
        bbox: BBox | None,
        time: TimeRange | None,
        levels: PressureLevels | None,
        extras: dict[str, Any],
    ) -> dict[str, Any]:
        form: dict[str, Any] = {
            "format": self.format,
            "product_type": self.product_type,
        }
        if variables:
            form["variable"] = self._encode_variables(variables)
        if bbox is not None:
            form["area"] = bbox.as_cds_area()
        if time is not None:
            form.update(time.as_cds_form())
        if levels is not None:
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
}

_FORMAT_ENGINE: dict[str, str | None] = {
    "netcdf": None,  # xarray picks the best netcdf engine
    "netcdf4": None,
    "nc": None,
    "grib": "cfgrib",
    "grib2": "cfgrib",
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
