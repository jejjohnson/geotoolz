"""`CogField` — a Cloud-Optimized GeoTIFF on object storage as a `Field`.

The reading engine is `geocloud.cog.CogSource` (geotoolz-cloud): the shared
obstore client pool, async-geotiff tile decoding, and batched reads that
fetch every tile a set of windows touches once. `CogField` adds the `Field`
spellings on top — ``select`` / ``select_many`` (and their ``aselect``
twins for `AsyncSpatialPatcher`), ``with_data`` — and the `PatchCache`
identity. `parallel_map` picks the batched ``select_many`` path
automatically.

Needs the ``[cog]`` extra: ``pip install 'geotoolz-patcher[cog]'``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from geopatcher._src._extras import missing_extra


if TYPE_CHECKING:
    from geocloud.cog import CogSource
else:
    try:
        from geocloud.cog import CogSource
    except ImportError:  # pragma: no cover - exercised on a slim install

        class CogSource:
            """Stand-in base without geotoolz-cloud: every use names the extra."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise missing_extra("CogField", "cog", "geotoolz-cloud[cog]")

            @classmethod
            def open(cls, *args: Any, **kwargs: Any) -> Any:
                raise missing_extra("CogField", "cog", "geotoolz-cloud[cog]")

            @classmethod
            async def aopen(cls, *args: Any, **kwargs: Any) -> Any:
                raise missing_extra("CogField", "cog", "geotoolz-cloud[cog]")


if TYPE_CHECKING:
    import numpy as np
    from georeader.geotensor import GeoTensor
    from rasterio.windows import Window


__all__ = ["CogField"]


class CogField(CogSource):
    """A tiled COG `Field` with batched range-fetch reads.

    Open with :meth:`open` (``await`` :meth:`aopen` inside an event loop);
    every argument is `geocloud.cog.CogSource.open`'s. Patches are
    `georeader.GeoTensor` chips with the window's transform, the CRS and
    the COG's nodata as ``fill_value_default``. The field pickles by URL,
    so it ships to process-pool workers.

    Examples:
        Patch a remote scene; `parallel_map` reads each chunk of patches
        with one batched fetch::

            field = CogField.open("s3://bucket/scene.tif")
            patches = list(patcher.split(field))
    """

    def select(self, window: Window) -> GeoTensor:
        """Read one window (`read_window`)."""
        return self.read_window(window)

    async def aselect(self, window: Window) -> GeoTensor:
        """Async twin of :meth:`select` (the `AsyncField` spelling)."""
        return await self.aread_window(window)

    def select_many(self, windows: list[Window]) -> list[GeoTensor]:
        """Read every window, fetching each overlapping tile once (`read_windows`)."""
        return self.read_windows(windows)

    async def aselect_many(self, windows: list[Window]) -> list[GeoTensor]:
        """Async twin of :meth:`select_many`."""
        return await self.aread_windows(windows)

    def with_data(self, array: np.ndarray) -> GeoTensor:
        """Wrap an operator output on the full grid as a `georeader.GeoTensor`.

        Carries the domain transform, CRS and nodata fill value.
        """
        from georeader.geotensor import GeoTensor

        return GeoTensor(
            values=array,
            transform=self.domain.transform,
            crs=self.domain.crs,
            fill_value_default=self.fill_value_default,
        )

    def cache_id(self) -> str:
        """`PatchCache` identity: the object read, its version and the IFD.

        `CogSource.identity` (the URL — or the explicit store's
        configuration plus ``path`` — and the IFD) plus
        `CogSource.object_version` (one ``HEAD`` per `split`: the ETag,
        else size + last-modified), so an object overwritten in place
        invalidates its cache entries.

        Raises:
            UnstableIdentityError: The explicit ``store`` has no printable
                configuration (``MemoryStore``); ``partial`` carries the
                rest of the identity (``path``, ``ifd_index``).
        """
        from geopatcher._src.cache import UnstableIdentityError

        identity: dict[str, Any] = {
            **self.identity(),
            "version": self.object_version(),
        }
        if self.store is not None and identity["store"] is None:
            raise UnstableIdentityError(
                f"its {type(self.store).__qualname__} has no configuration that "
                f"names its contents.",
                partial=json.dumps(identity, sort_keys=True),
            )
        return json.dumps(identity, sort_keys=True)
