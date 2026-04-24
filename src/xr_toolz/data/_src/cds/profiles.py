"""Per-dataset-family form profiles for the CDS adapter.

The CDS API speaks several dialects that share a ``cdsapi.Client.retrieve``
envelope but differ in which form keys are accepted. The schemas below
were dumped from the live CDS ``/retrieve/v1/processes/<dataset_id>``
endpoint — not inferred from the docs — so they reflect what the API
actually accepts today.

- **Reanalysis-style** (ERA5 single/pressure levels, ERA5-Land) uses
  ``format``, ``product_type``, ``area``, ``year/month/day``
  (all arrays), and optionally ``pressure_level``.
- **In-situ surface-land** uses ``data_format`` (``csv`` or
  ``netcdf``), ``version``, ``time_aggregation``, ``variable``,
  ``area``, a **single-valued** ``year`` string, and array
  ``month`` / ``day``. No ``product_type``.
- **In-situ surface-marine** is like land *without*
  ``time_aggregation`` — marine archives ship as a single aggregation
  tier baked into the product.

A :class:`CDSFormProfile` makes these differences explicit so the
single :class:`~xr_toolz.data._src.cds.source.CDSSource` can target all
three without branching on dataset id in the adapter. Each
:class:`~xr_toolz.data._src.base.DatasetInfo` entry for a CDS dataset
carries a ``form_profile`` field that the adapter consults when
building the request form.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CDSFormProfile:
    """Shape of a CDS ``retrieve()`` form for a dataset family.

    Attributes:
        family: Short slug identifying the family (``"reanalysis"``,
            ``"insitu-land"``, ``"insitu-marine"``, ...). Used in logs
            / tests.
        format_key: Name of the form field that carries the output
            format. ``"format"`` for reanalysis; ``"data_format"`` for
            CDS in-situ (the API uses different names).
        format_default: Default value for ``format_key`` when the
            caller / source don't set it. ``"netcdf"`` for reanalysis;
            ``"csv"`` for in-situ.
        fixed: Form keys this family always sends with a fixed value
            (e.g. ``{"version": "2_0_0"}`` for in-situ). Overridable
            via ``**extras``.
        includes_product_type: Whether ``product_type`` is part of the
            form.
        uses_area: Whether ``bbox`` serialises to ``area``.
        uses_pressure_level: Whether :class:`PressureLevels` serialises
            to ``pressure_level``.
        year_is_array: Whether ``year`` accepts a list. In-situ
            products take ``year`` as a single string — ``time`` must
            be within one calendar year per request. The
            :class:`~xr_toolz.data._src.cds.archive.CDSInsituArchive`
            chunks by year to respect this.
        required_extras: Keys the caller must supply via ``**extras``
            (e.g. ``"time_aggregation"`` for in-situ-land).
    """

    family: str
    format_key: str = "format"
    format_default: str = "netcdf"
    fixed: Mapping[str, Any] = field(default_factory=dict, hash=False)
    includes_product_type: bool = False
    uses_area: bool = True
    uses_pressure_level: bool = False
    year_is_array: bool = True
    required_extras: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Freeze ``fixed`` the same way ``Variable.aliases`` is frozen.
        if not isinstance(self.fixed, MappingProxyType):
            frozen = MappingProxyType(dict(self.fixed))
            object.__setattr__(self, "fixed", frozen)


REANALYSIS = CDSFormProfile(
    family="reanalysis",
    format_key="format",
    format_default="netcdf",
    includes_product_type=True,
    uses_pressure_level=True,
)
"""Profile for ERA5 / ERA5-Land style gridded reanalyses.

``format=netcdf``, ``product_type=reanalysis`` (source-configurable),
``area``, ``year/month/day`` (all arrays), and optional
``pressure_level``.
"""


INSITU_LAND = CDSFormProfile(
    family="insitu-land",
    format_key="data_format",
    format_default="csv",
    fixed={"version": "2_0_0"},
    includes_product_type=False,
    uses_area=True,
    year_is_array=False,
    required_extras=("time_aggregation",),
)
"""Profile for ``insitu-observations-surface-land``.

CSV output, ``version=2_0_0``, single-valued ``year``, arrays for
``month`` / ``day``, ``area`` accepted, and caller must supply
``time_aggregation`` ∈ ``{sub_daily, daily, monthly}`` via ``**extras``.
"""


INSITU_MARINE = CDSFormProfile(
    family="insitu-marine",
    format_key="data_format",
    format_default="csv",
    fixed={"version": "2_0_0"},
    includes_product_type=False,
    uses_area=True,
    year_is_array=False,
)
"""Profile for ``insitu-observations-surface-marine``.

Same shape as :data:`INSITU_LAND` but without ``time_aggregation`` —
marine ships a single aggregation tier baked into the product.
"""


# Back-compat alias — some external callers may have already imported
# ``INSITU`` from a prior iteration of this module. It aliases the
# surface-land profile, which was the "primary" in-situ target.
INSITU = INSITU_LAND


def resolve_profile(
    dataset_id: str, datasets: dict[str, Any] | None = None
) -> CDSFormProfile:
    """Return the :class:`CDSFormProfile` for ``dataset_id``.

    ``datasets`` defaults to :data:`CDS_DATASETS` (imported lazily to
    avoid a circular import). Unknown dataset ids fall back to
    :data:`REANALYSIS` so existing call sites that don't use the
    catalog stay bit-compatible.
    """
    if datasets is None:
        from xr_toolz.data._src.cds.catalog import CDS_DATASETS

        datasets = CDS_DATASETS
    info = datasets.get(dataset_id)
    if info is None:
        return REANALYSIS
    profile: CDSFormProfile | None = getattr(info, "form_profile", None)
    return profile or REANALYSIS
