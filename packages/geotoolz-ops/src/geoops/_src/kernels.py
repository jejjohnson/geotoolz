"""Kernel operators: one element-wise L1 kernel, run on every carrier.

A kernel operator reads a few labelled bands of one dim, hands them to an
element-wise Tier-A kernel through pipekit-array's `KernelOp`, and writes
``NaN`` where any band it read is nodata. The same call runs on

- a plain NumPy array, read in the 0.2 layout ``DIMS[-ndim:]``;
- a NumPy or JAX `geoarray.GeoArray` (under ``jax.jit`` too);
- an ``xr.DataArray``, lazily through ``apply_ufunc`` when it is dask-backed.

The kernel sees arrays only, so it is written once against the array
namespace (``array_namespace(x)``) and never imports a carrier.

Two rules keep the results what the NumPy operators returned:

- **Nodata.** A pixel is nodata when a band the kernel reads is non-finite
  or equals the carrier's ``fill_value`` (`geoops._src.valid`). Its value
  is ``NaN``, and a `GeoArray` output declares ``fill_value=NaN``.
- **Time stacks.** A band-collapsing result on a ``(time, band, y, x)``
  `GeoArray` or 4-D plain array keeps a singleton band dim,
  ``(T, 1, H, W)``, as `geoops._src.shape.keep_band_axis` does. A
  ``DataArray`` keeps its natural ``(time, y, x)``: its dims are named.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from geoarray import GeoArray, GridMismatchError
from pipekit import get_adapter
from pipekit_array import DimArray, KernelOp, array_namespace, isel

from geoops._src.bands import BandRef, resolve_band
from geoops._src.geo import reject_geotensor
from geoops._src.shape import DIMS


__all__ = [
    "band_kernel",
    "carrier_kernel",
    "fill_of",
    "mask_kernel",
    "named",
    "nodata_where",
    "select_band",
]


def named(x: Any, *, name: str) -> tuple[Any, Callable[[Any], Any]]:
    """``x`` as a named carrier, and the function that turns a result back.

    A registered carrier (`GeoArray`, ``DataArray``, `DimArray`) passes
    through. A plain array becomes a `DimArray` with the 0.2 dims
    ``DIMS[-ndim:]``, and the result comes back as its raw array.

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``.
        ValueError: A plain array is not 2-D to 4-D.
    """
    reject_geotensor(x, name)
    if get_adapter(x) is not None:
        return x, lambda out: out
    data = np.asarray(x) if not hasattr(x, "shape") else x
    ndim = len(data.shape)
    if not 2 <= ndim <= len(DIMS):
        raise ValueError(f"{name}: a plain array must be 2-D to 4-D, got {ndim}-D")
    return DimArray(data, DIMS[-ndim:]), _raw


def _raw(out: Any) -> Any:
    return out.data if isinstance(out, DimArray) else out


def _dims(x: Any) -> tuple[str, ...]:
    adapter = get_adapter(x)
    if adapter is None:
        raise TypeError(f"{type(x).__name__} is not a named carrier")
    return tuple(adapter.dims(x))


def select_band(x: Any, dim: str, band: BandRef, *, name: str) -> Any:
    """One band of ``x`` along ``dim``, by position or by name; ``dim`` dropped.

    A name is looked up in the carrier's labels for ``dim`` (a `GeoArray`'s
    `SpectralAxis`, a ``DataArray``'s index coordinate), then in its band
    attrs (`geoops._src.bands.resolve_band`).

    Raises:
        ValueError: ``x`` has no ``dim``; the message lists its dims.
        TypeError: ``band`` is a name and ``x`` names no bands.
    """
    dims = _dims(x)
    if dim not in dims:
        raise ValueError(f"{name}: the carrier has no {dim!r} dim; its dims are {dims}")
    return isel(x, **{dim: _position(x, dim, band)})


def _position(x: Any, dim: str, band: BandRef) -> int:
    if isinstance(band, str):
        adapter = get_adapter(x)
        labels = adapter.labels(x, dim) if adapter is not None else None
        if labels is not None and band in labels:
            return labels.index(band)
    if isinstance(x, DimArray):
        # a plain array: names have nothing to resolve against
        return resolve_band(x.data, band)
    return resolve_band(x, band)


def nodata_where(value: Any, *bands: Any, fill_values: Sequence[Any]) -> Any:
    """``value`` where every band is valid, ``NaN`` elsewhere.

    A band is invalid where it is non-finite or equals its fill (a ``NaN``
    fill is covered by the finiteness test). Written against the array
    namespace, so it runs on NumPy, JAX and dask blocks.

    Args:
        value: The kernel's result, floating point.
        *bands: The inputs the kernel read, on ``value``'s shape.
        fill_values: One fill per band, ``None`` for none.

    Returns:
        ``value`` with ``NaN`` in its nodata pixels.

    Example:
        >>> a = np.array([1.0, np.nan, -9.0])
        >>> nodata_where(a * 2, a, fill_values=(-9.0,))
        array([ 2., nan, nan])
    """
    xp = array_namespace(value)
    valid = None
    for band, fill in zip(bands, fill_values, strict=True):
        ok = xp.isfinite(band)
        if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
            ok = ok & (band != fill)
        valid = ok if valid is None else valid & ok
    if valid is None:
        return value
    return xp.where(valid, value, xp.nan)


def _masked_kernel(
    *bands: Any,
    kernel: Callable[..., Any],
    fill_values: tuple[Any, ...],
    params: dict[str, Any],
) -> Any:
    return nodata_where(kernel(*bands, **params), *bands, fill_values=fill_values)


def fill_of(x: Any) -> Any:
    """The fill that marks nodata in ``x``, or ``None``.

    A `GeoArray` declares ``fill_value``. A ``DataArray`` declares it in
    ``attrs["fill_value"]`` (written by `GeoArray.to_xarray`), else in an
    undecoded ``attrs["_FillValue"]``; a file xarray decoded already holds
    ``NaN`` there. Anything else declares none.
    """
    if isinstance(x, GeoArray):
        return x.fill_value
    if _is_dataarray(x):
        return x.attrs.get("fill_value", x.attrs.get("_FillValue"))
    return None


def band_kernel(
    kernel: Callable[..., Any],
    x: Any,
    bands: Sequence[BandRef],
    *,
    dim: str,
    name: str,
    **params: Any,
) -> Any:
    """Run an element-wise kernel on bands of ``x``; nodata becomes ``NaN``.

    Args:
        kernel: The Tier-A kernel: one array per band, then ``**params``;
            element-wise (signature ``"(),...,()->()"``).
        x: The carrier: a `GeoArray`, a ``DataArray``, a `DimArray` or a
            plain 2-D to 4-D array.
        bands: The bands the kernel reads, in its argument order.
        dim: The band dim.
        name: The operator's name, for messages.
        **params: Keyword parameters of the kernel (``eps``, ``L``, …).

    Returns:
        The kernel's result on ``x``'s carrier with ``dim`` collapsed: a
        `GeoArray` declaring ``fill_value=NaN``, a ``DataArray``, or a plain
        array for a plain array. A ``(time, band, y, x)`` `GeoArray` or 4-D
        plain array keeps a singleton band dim.

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``, or a band name meets
            a carrier without band names.
        ValueError: ``x`` has no ``dim``, or a band name is unknown.

    Example:
        >>> from geoops.indices import normalized_difference
        >>> scene = np.stack([np.full((2, 2), 0.1), np.full((2, 2), 0.5)])
        >>> band_kernel(
        ...     normalized_difference, scene, (1, 0), dim="band", name="ND", eps=0.0
        ... ).round(3)
        array([[0.667, 0.667],
               [0.667, 0.667]])
    """
    carrier, unwrap = named(x, name=name)
    dims = _dims(carrier)
    selected = [select_band(carrier, dim, band, name=name) for band in bands]
    out = _run(kernel, selected, [fill_of(carrier)] * len(selected), params)
    if "time" in dims and len(dims) == 4 and not _is_dataarray(carrier):
        out = _expand(out, dims.index(dim), dim)
    return unwrap(_declare_fill(out, np.nan))


def carrier_kernel(
    kernel: Callable[..., Any], *xs: Any, name: str, **params: Any
) -> Any:
    """Run an element-wise kernel on whole carriers of one grid; nodata becomes ``NaN``.

    Each carrier's own ``fill_value`` marks its nodata. Carriers of one
    family are checked to share a grid by their adapter (a `GeoArray`
    raises `geoarray.GridMismatchError`, a `ValueError`).

    Args:
        kernel: The Tier-A kernel: one array per carrier, then ``**params``.
        *xs: The carriers, all `GeoArray`, all ``DataArray`` or all plain
            arrays of one shape.
        name: The operator's name, for messages.
        **params: Keyword parameters of the kernel.

    Returns:
        The result on the first carrier's carrier type, a `GeoArray`
        declaring ``fill_value=NaN``.

    Raises:
        TypeError: A carrier is a georeader ``GeoTensor``.
        ValueError: The carriers do not share a grid (or, plain, a shape); the
            message says they must share shape, transform, and CRS.

    Example:
        >>> from geoops.indices import dnbr
        >>> carrier_kernel(dnbr, np.ones((2, 2)), np.zeros((2, 2)), name="dNBR")
        array([[1., 1.],
               [1., 1.]])
    """
    pairs = [named(x, name=name) for x in xs]
    carriers = [c for c, _ in pairs]
    message = f"{name} inputs must share shape, transform, and CRS"
    shapes = {tuple(c.shape) for c in carriers}
    if len(shapes) > 1:
        raise ValueError(f"{message}; got shapes {sorted(shapes)}")
    try:
        out = _run(kernel, carriers, [fill_of(c) for c in carriers], params)
    except GridMismatchError as exc:
        raise GridMismatchError(f"{message}: {exc}") from exc
    return pairs[0][1](_declare_fill(out, np.nan))


def mask_kernel(
    kernel: Callable[..., Any],
    x: Any,
    band: BandRef | None,
    *,
    dim: str,
    name: str,
    **params: Any,
) -> Any:
    """Run an element-wise mask kernel on one band of ``x``, or on all of it.

    For decoders that read their own input (a QA band): no nodata is
    judged, and the boolean result declares ``fill_value=False``.

    Args:
        kernel: The Tier-A kernel: one array, then ``**params``; returns a
            boolean array of its shape.
        x: The carrier: a `GeoArray`, a ``DataArray``, a `DimArray` or a
            plain 2-D to 4-D array.
        band: The band to decode along ``dim``, or ``None`` to decode every
            element of ``x``.
        dim: The band dim.
        name: The operator's name, for messages.
        **params: Keyword parameters of the kernel (``bits``, ``values``, …).

    Returns:
        The mask on ``x``'s carrier: ``x``'s dims when ``band`` is ``None``,
        else ``dim`` collapsed (a singleton band dim on a ``(T, C, H, W)``
        `GeoArray` or plain array).

    Raises:
        TypeError: ``x`` is a georeader ``GeoTensor``, or a band name meets
            a carrier without band names.
        ValueError: ``band`` is given and ``x`` has no ``dim``.

    Example:
        >>> from geoops.qa import mask_from_qa_bits
        >>> qa = np.array([[0b1000, 0b0001]], dtype=np.uint16)
        >>> mask_kernel(mask_from_qa_bits, qa, None, dim="band", name="Q", bits=(3,))
        array([[ True, False]])
    """
    carrier, unwrap = named(x, name=name)
    dims = _dims(carrier)
    picked = carrier if band is None else select_band(carrier, dim, band, name=name)
    out = KernelOp(kernel, "()->()", core={}, params=params)(picked)
    if (
        band is not None
        and "time" in dims
        and len(dims) == 4
        and not _is_dataarray(carrier)
    ):
        out = _expand(out, dims.index(dim), dim)
    return unwrap(_declare_fill(out, False))


def _run(
    kernel: Callable[..., Any],
    inputs: Sequence[Any],
    fills: Sequence[Any],
    params: dict[str, Any],
) -> Any:
    spec = ",".join("()" for _ in inputs) + "->()"
    op = KernelOp(
        _masked_kernel,
        spec,
        core={},
        params={"kernel": kernel, "fill_values": tuple(fills), "params": params},
    )
    return op(*inputs)


def _declare_fill(out: Any, fill: Any) -> Any:
    """Declare ``fill`` as the output's fill, where the carrier declares one."""
    if isinstance(out, GeoArray):
        return out.replace(fill_value=fill)
    if _is_dataarray(out) and {"fill_value", "_FillValue"} & set(out.attrs):
        out = out.copy(deep=False)
        out.attrs = {k: v for k, v in out.attrs.items() if k != "_FillValue"}
        out.attrs["fill_value"] = fill
    return out


def _is_dataarray(x: Any) -> bool:
    return type(x).__module__.startswith("xarray.")


def _expand(out: Any, axis: int, dim: str) -> Any:
    if isinstance(out, GeoArray):
        return out.expand_dims(axis, dim)
    data = out.data
    xp = array_namespace(data)
    dims = (*out.dims[:axis], dim, *out.dims[axis:])
    return DimArray(xp.expand_dims(data, axis=axis), dims)
