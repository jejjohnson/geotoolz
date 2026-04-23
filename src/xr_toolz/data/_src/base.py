"""Abstract base classes for data sources and dataset descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import xarray as xr

from xr_toolz.types import (
    BBox,
    DepthRange,
    PressureLevels,
    TimeRange,
    Variable,
    resolve,
)


class DatasetKind(StrEnum):
    """Discriminator for how a dataset is laid out once materialized.

    - ``gridded`` — regular lon/lat (+time) cube, the common case.
    - ``alongtrack`` — 1-D samples along a satellite ground track.
    - ``profiles`` — collection of vertical profiles at scattered
      locations (Argo floats, CTD casts, gliders).
    - ``trajectory`` — drifters / moving platforms.

    Downstream code (regridding, masking, plotting) branches on this;
    e.g. antimeridian BBox wrapping is meaningless for profiles.
    """

    GRIDDED = "gridded"
    ALONGTRACK = "alongtrack"
    PROFILES = "profiles"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class DatasetInfo:
    """Static description of a remote dataset.

    Adapters return these from :meth:`DataSource.describe` and the
    catalog stores them alongside default request parameters.
    """

    dataset_id: str
    source: str
    title: str
    kind: DatasetKind = DatasetKind.GRIDDED
    variables: tuple[Variable, ...] = ()
    spatial_coverage: BBox | None = None
    temporal_coverage: tuple[str, str] | None = None
    doi: str | None = None
    license: str | None = None
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract interface implemented by every adapter."""

    source_id: str
    """Short slug used to look up per-source aliases in :class:`Variable`."""

    @abstractmethod
    def list_datasets(self) -> list[DatasetInfo]:  # pragma: no cover - interface
        """Return known datasets exposed by this source."""

    @abstractmethod
    def describe(self, dataset_id: str) -> DatasetInfo:  # pragma: no cover - interface
        """Return a :class:`DatasetInfo` for ``dataset_id``."""

    @abstractmethod
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
    ) -> Path:  # pragma: no cover - interface
        """Download ``dataset_id`` to ``output`` and return the path."""

    @abstractmethod
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
    ) -> xr.Dataset:  # pragma: no cover - interface
        """Open ``dataset_id`` lazily as an :class:`xarray.Dataset`."""

    # ---- helpers --------------------------------------------------------

    def _resolve_variables(
        self, variables: list[str | Variable] | None
    ) -> list[Variable]:
        """Translate a heterogeneous variable list into :class:`Variable`."""
        if variables is None:
            return []
        return [resolve(v) for v in variables]

    def _encode_variables(self, variables: list[str | Variable] | None) -> list[str]:
        """Return source-specific identifiers for ``variables``."""
        resolved = self._resolve_variables(variables)
        return [v.for_source(self.source_id) for v in resolved]
