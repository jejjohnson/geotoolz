"""Copernicus Marine (CMEMS) adapter built on ``copernicusmarine``.

The underlying client is imported lazily so ``xr_toolz.data`` can be
imported without the optional ``copernicusmarine`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from xr_toolz.data._src.base import DatasetInfo, DataSource
from xr_toolz.data._src.cmems.catalog import CMEMS_DATASETS
from xr_toolz.data._src.credentials import CMEMSCredentials, load_cmems
from xr_toolz.types import (
    BBox,
    DepthRange,
    PressureLevels,
    TimeRange,
    Variable,
)


class CMEMSSource(DataSource):
    """Adapter around the ``copernicusmarine`` Python client.

    Args:
        credentials: Explicit :class:`CMEMSCredentials`. When ``None``,
            credentials are resolved from env vars / ``~/.cmems``.
        client: Optional pre-built client (``copernicusmarine`` module
            or test double). Useful for tests; production code leaves
            this ``None``.
    """

    source_id = "cmems"

    def __init__(
        self,
        credentials: CMEMSCredentials | None = None,
        client: Any | None = None,
    ) -> None:
        self.credentials = credentials or load_cmems()
        self._client = client

    # ---- client handling --------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import copernicusmarine  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "CMEMSSource requires the 'copernicusmarine' package. "
                "Install with: pip install xr_toolz[data]"
            ) from exc
        self._client = copernicusmarine
        return self._client

    def _auth_kwargs(self) -> dict[str, str]:
        if self.credentials is None:
            return {}
        return {
            "username": self.credentials.username,
            "password": self.credentials.password,
        }

    # ---- DataSource API ---------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        return list(CMEMS_DATASETS.values())

    def describe(self, dataset_id: str) -> DatasetInfo:
        if dataset_id in CMEMS_DATASETS:
            return CMEMS_DATASETS[dataset_id]
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
        """Download a subset to a NetCDF file at ``output``."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        kwargs = self._subset_kwargs(
            dataset_id=dataset_id,
            variables=variables,
            bbox=bbox,
            time=time,
            depth=depth,
            extras=extras,
        )
        self._get_client().subset(
            **kwargs,
            output_filename=output.name,
            output_directory=str(output.parent),
            **self._auth_kwargs(),
        )
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
        """Open a lazy dataset via ``copernicusmarine.open_dataset``."""
        kwargs = self._subset_kwargs(
            dataset_id=dataset_id,
            variables=variables,
            bbox=bbox,
            time=time,
            depth=depth,
            extras=extras,
        )
        return self._get_client().open_dataset(**kwargs, **self._auth_kwargs())

    # ---- payload construction --------------------------------------------

    def _subset_kwargs(
        self,
        dataset_id: str,
        variables: list[str | Variable] | None,
        bbox: BBox | None,
        time: TimeRange | None,
        depth: DepthRange | None,
        extras: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"dataset_id": dataset_id}
        if variables:
            payload["variables"] = self._encode_variables(variables)
        if bbox is not None:
            payload.update(bbox.as_cmems())
        if time is not None:
            payload.update(time.as_cmems())
        if depth is not None:
            payload.update(depth.as_cmems())
        payload.update(extras)
        return payload
