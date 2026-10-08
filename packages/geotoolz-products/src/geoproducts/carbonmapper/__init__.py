"""Carbon Mapper reader subpackage.

Reader for the `Carbon Mapper <https://carbonmapper.org>`_ STAC + plume
catalogue API, split into modules by concern:

- :mod:`~geoproducts.carbonmapper.config` —
  :class:`CarbonMapperConfig` token persistence (env vars, file, or
  in-memory). Pure file-based; no Azure SDK dependency.
- :mod:`~geoproducts.carbonmapper.download` — raw HTTP / JSON
  primitives (``obtain_token``, ``stac_get_items``,
  ``download_asset``, etc.). Pulls in :mod:`requests`.
- :mod:`~geoproducts.carbonmapper.api_queries` — typed wrappers
  on top of ``download``: :class:`CMTileItem`, :func:`get_plume`,
  :func:`list_tiles`, exception hierarchy, etc. Returns
  :class:`CMRawPlume` / :class:`CMSource` / :class:`CMTileItem`
  instances — never raw dicts.
- :mod:`~geoproducts.carbonmapper.plume` —
  :class:`CMRawPlume` Pydantic model accepting both CSV bulk-export
  and annotated-JSON payloads.
- :mod:`~geoproducts.carbonmapper.source` — :class:`CMSource`
  dataclass for ``/catalog/sources.geojson`` features.
- :mod:`~geoproducts.carbonmapper.rasters` —
  :class:`CMImageRaster` (L2B scene). Lazy
  ``RasterioReader``-backed band accessors that delegate to
  ``georeader.read``.
- :mod:`~geoproducts.carbonmapper.image` —
  :class:`CMPlumeImage` (per-plume L3A product bundle: mask,
  concentrations, IME-clipped concentrations, RGB, outline). Handles
  both v3a (STAC-resident) and v3c (CDN-only) plumes via
  URL-pattern derivation.

Optional install: ``pip install 'geotoolz-products[carbonmapper]'``
(adds ``pydantic`` and ``requests`` — both Carbon-Mapper-only).

References
----------
- Product Guide: https://carbonmapper.org/articles/product-guide
- API Docs:      https://api.carbonmapper.org/api/v1/docs
- STAC Catalog:  https://api.carbonmapper.org/api/v1/stac
"""

import importlib.util as _importlib_util

from geoproducts._src.extras import missing_extra as _missing_extra


# This subpackage is the ``[carbonmapper]`` extra's entry point: fail at
# import, naming the extra, rather than deep inside a submodule.
for _module in ("requests", "pydantic"):
    if _importlib_util.find_spec(_module) is None:
        raise _missing_extra("geoproducts.carbonmapper", "carbonmapper")

from geoproducts.carbonmapper.api_queries import (
    DEFAULT_L2B_COLLECTION,
    CMAPIError,
    CMPlumeNotFound,
    CMSceneNotPublished,
    CMSourceNotFound,
    CMTileItem,
    get_plume,
    get_plume_context,
    get_source,
    get_source_for_plume,
    get_tile,
    get_tile_for_plume,
    list_plumes,
    list_plumes_for_source,
    list_plumes_for_tile,
    list_sources,
    list_tiles,
    list_tiles_for_source,
)
from geoproducts.carbonmapper.config import CarbonMapperConfig
from geoproducts.carbonmapper.download import (
    download_asset,
    download_plume_assets,
    export_plumes_to_geojson,
    get_plume_by_id,
    get_plumes_annotated,
    get_plumes_csv,
    get_scenes,
    get_sources,
    obtain_token,
    paginate_plumes,
    refresh_token,
    stac_get_catalog,
    stac_get_collection,
    stac_get_items,
    stac_list_collections,
    stac_search,
)
from geoproducts.carbonmapper.image import (
    CM_PLUME_IMAGE_ASSETS,
    CMPlumeImage,
)
from geoproducts.carbonmapper.plume import (
    CARBONMAPPER_INSTRUMENTS,
    CARBONMAPPER_PLUME_PARAMS,
    CM_INSTRUMENT_TO_SATELLITE,
    CMRawPlume,
    Collection,
    Gas,
    Instrument,
    decompose_wind,
)
from geoproducts.carbonmapper.products import (
    ALL_PLUME_PRODUCTS,
    ALL_PRODUCTS,
    DEFAULT_PLUME_PRODUCTS,
    DEFAULT_SCENE_PRODUCTS,
    CMCollectionSpec,
    CMProduct,
    CMProductFamily,
    CMProductNotSelected,
    CMQuicklookProduct,
    CMRasterProduct,
    CMTextProduct,
    CMVectorProduct,
    product_for_key,
)
from geoproducts.carbonmapper.rasters import (
    CM_L2B_BANDS,
    DEFAULT_L2B_RGB_COLLECTION,
    CMImageRaster,
)
from geoproducts.carbonmapper.source import CMSource
from geoproducts.carbonmapper.sources_raster import (
    CMSourceRaster,
    rasterize_sources,
    rasterize_sources_like,
)


__all__ = [
    "ALL_PLUME_PRODUCTS",
    "ALL_PRODUCTS",
    "CARBONMAPPER_INSTRUMENTS",
    "CARBONMAPPER_PLUME_PARAMS",
    "CM_INSTRUMENT_TO_SATELLITE",
    "CM_L2B_BANDS",
    "CM_PLUME_IMAGE_ASSETS",
    "DEFAULT_L2B_COLLECTION",
    "DEFAULT_L2B_RGB_COLLECTION",
    "DEFAULT_PLUME_PRODUCTS",
    "DEFAULT_SCENE_PRODUCTS",
    "CMAPIError",
    "CMCollectionSpec",
    "CMImageRaster",
    "CMPlumeImage",
    "CMPlumeNotFound",
    "CMProduct",
    "CMProductFamily",
    "CMProductNotSelected",
    "CMQuicklookProduct",
    "CMRasterProduct",
    "CMRawPlume",
    "CMSceneNotPublished",
    "CMSource",
    "CMSourceNotFound",
    "CMSourceRaster",
    "CMTextProduct",
    "CMTileItem",
    "CMVectorProduct",
    "CarbonMapperConfig",
    "Collection",
    "Gas",
    "Instrument",
    "decompose_wind",
    "download_asset",
    "download_plume_assets",
    "export_plumes_to_geojson",
    "get_plume",
    "get_plume_by_id",
    "get_plume_context",
    "get_plumes_annotated",
    "get_plumes_csv",
    "get_scenes",
    "get_source",
    "get_source_for_plume",
    "get_sources",
    "get_tile",
    "get_tile_for_plume",
    "list_plumes",
    "list_plumes_for_source",
    "list_plumes_for_tile",
    "list_sources",
    "list_tiles",
    "list_tiles_for_source",
    "obtain_token",
    "paginate_plumes",
    "product_for_key",
    "rasterize_sources",
    "rasterize_sources_like",
    "refresh_token",
    "stac_get_catalog",
    "stac_get_collection",
    "stac_get_items",
    "stac_list_collections",
    "stac_search",
]
