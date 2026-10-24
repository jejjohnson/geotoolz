"""Readable types: a name, a dims pattern and a geometry kind (GA-T5, CS-F1).

A readable type is not a subclass. ``GeoImage`` is "a grid carrier with
dims ``band y x``", and any carrier matches it when its pipekit adapter
reports a fitting `pipekit.Signature` and its *kind reader* says it is a
grid: a `GeoArray`, or (with ``geotoolz-array[xarray]``) a lazy
``DataArray`` cube whose horizontal dims are ``lat lon``. Slicing a
``GeoImageSeries`` at one date simply is a ``GeoImage``.

- `ReadableType` — the description; `matches(x, T)` the test.
- The aliases (``GeoMap``, ``GeoImage``, …) are
  ``Annotated[<carrier>, ReadableType(...)]``, so they read as the carrier
  to a type checker and carry the pattern at run time.
- `check_dims` checks annotated arguments and the return value at the call.
- `register_kind_reader` — how a carrier family says which kind it is and
  which of its dims are the horizontal ones (read as ``y x``, ``row col``
  or ``obs``). Kind is a geo idea, so it lives here, not in pipekit.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import types as _types
import typing
from collections.abc import Callable
from typing import Annotated, Any, NamedTuple

from pipekit import Signature, get_adapter

from geoarray._src.errors import GeoArrayError


__all__ = [
    "DimsError",
    "Kind",
    "ReadableType",
    "as_readable",
    "check_dims",
    "kind_of",
    "matches",
    "register_kind_reader",
]

#: the horizontal dims a pattern names, per kind
CANONICAL: dict[str, tuple[str, ...]] = {
    "grid": ("y", "x"),
    "swath": ("row", "col"),
    "points": ("obs",),
}


class DimsError(GeoArrayError, TypeError):
    """A carrier's dims (or kind) do not fit the readable type a function declares."""


class Kind(NamedTuple):
    """What a kind reader says about a carrier.

    Attributes:
        kind: ``"grid"``, ``"swath"`` or ``"points"``.
        horizontal: The carrier's own names for the horizontal dims
            (``("lat", "lon")`` on a CF cube), read as the kind's canonical
            ones in patterns.
        leaves: The optional geometry a point set carries (``"z"``,
            ``"time"``), which `GeoProfiles` / `GeoTracks` require.
    """

    kind: str
    horizontal: tuple[str, ...]
    leaves: frozenset[str] = frozenset()


KindReader = Callable[[Any], Kind | None]

_READERS: dict[type, KindReader] = {}
#: module prefix of a carrier type -> a loader registering its reader
_LAZY: dict[str, Callable[[], None]] = {}


def register_kind_reader(carrier_type: type, reader: KindReader) -> None:
    """Register how carriers of ``carrier_type`` (and subclasses) read as a kind.

    ``reader(x)`` returns a `Kind`, or ``None`` when this ``x`` has no
    georeference a readable type can name.
    """
    _READERS[carrier_type] = reader


def register_lazy_kind_reader(module_prefix: str, loader: Callable[[], None]) -> None:
    """Call ``loader`` the first time a carrier from ``module_prefix`` is read."""
    _LAZY[module_prefix] = loader


def kind_of(x: Any) -> Kind | None:
    """The `Kind` of ``x``, or ``None`` (a raw array, a `DimArray`).

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 2)), (1, 0, 0, 0, -1, 0), None, dims="y x")
        >>> kind_of(g)
        Kind(kind='grid', horizontal=('y', 'x'), leaves=frozenset())
    """
    cls = type(x)
    reader = _lookup(cls)
    if reader is None:
        for prefix in list(_LAZY):
            if cls.__module__.startswith(prefix):
                _LAZY.pop(prefix)()
                reader = _lookup(cls)
                break
    return None if reader is None else reader(x)


def _lookup(cls: type) -> KindReader | None:
    for base in cls.__mro__:
        if base in _READERS:
            return _READERS[base]
    return None


@dataclasses.dataclass(frozen=True)
class ReadableType:
    """A name for a common carrier shape.

    Args:
        name: ``"GeoImage"``.
        pattern: The dims in canonical names: ``"band y x"`` for a grid,
            ``"band row col"`` for a swath, ``"variable obs"`` for points;
            ``...`` for any dims (``"... y x"``).
        kind: ``"grid"``, ``"swath"`` or ``"points"``.
        leaves: Geometry a point set must carry: ``("time",)`` for tracks,
            ``("z",)`` for profiles.

    Example:
        >>> GeoImage = ReadableType("GeoImage", "band y x", "grid")
        >>> GeoImage
        GeoImage[grid: band y x]
    """

    name: str
    pattern: str
    kind: str
    leaves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in CANONICAL:
            raise ValueError(f"kind is one of {sorted(CANONICAL)}, got {self.kind!r}")
        horizontal = CANONICAL[self.kind]
        names = self.pattern.split()
        if tuple(names[len(names) - len(horizontal) :]) != horizontal:
            raise ValueError(
                f"a {self.kind} pattern ends in {' '.join(horizontal)!r}, got "
                f"{self.pattern!r}"
            )

    @property
    def dims(self) -> tuple[str, ...]:
        """The pattern's names (``...`` included)."""
        return tuple(self.pattern.split())

    def __repr__(self) -> str:
        extra = f" + {', '.join(self.leaves)}" if self.leaves else ""
        return f"{self.name}[{self.kind}: {self.pattern}{extra}]"


def _canonical_signature(x: Any, kind: Kind) -> Signature | None:
    adapter = get_adapter(x)
    if adapter is None:
        return None
    sig = adapter.signature(x)
    rename = dict(zip(kind.horizontal, CANONICAL[kind.kind], strict=True))
    return Signature(dims=tuple((rename.get(d, d), n) for d, n in sig.dims))


def _mismatch(x: Any, t: ReadableType) -> str | None:
    """Why ``x`` is not a ``t``, or ``None`` when it is."""
    kind = kind_of(x)
    if kind is None:
        return f"a {type(x).__name__} has no georeference"
    if kind.kind != t.kind:
        return f"a {type(x).__name__} is a {kind.kind}, not a {t.kind}"
    sig = _canonical_signature(x, kind)
    if sig is None or not sig.matches(t.pattern):
        dims = None if sig is None else tuple(d for d, _ in sig.dims)
        return f"dims {dims}"
    missing = [leaf for leaf in t.leaves if leaf not in kind.leaves]
    if missing:
        return f"no {', '.join(missing)} on the samples"
    return None


def matches(x: Any, t: Any) -> bool:
    """Whether carrier ``x`` is a ``t``: its kind, dims and required leaves.

    Dims are read through ``x``'s pipekit adapter, with its horizontal dims
    renamed to the canonical ones (a CF cube's ``lat lon`` read as ``y x``);
    nothing is renamed on ``x``.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> from geoarray.types import GeoImage, GeoMap
        >>> stack = GeoArray(np.zeros((3, 2, 4, 4)), (1, 0, 0, 0, -1, 0), None,
        ...                  dims="time band y x")
        >>> matches(stack.isel(time=0), GeoImage), matches(stack, GeoMap)
        (True, False)
    """
    return _mismatch(x, as_readable(t)) is None


def as_readable(t: Any) -> ReadableType:
    """``t`` as a `ReadableType`: itself, or the one an alias carries.

    Raises:
        TypeError: ``t`` is neither.
    """
    if isinstance(t, ReadableType):
        return t
    spec = spec_of(t)
    if spec is None:
        raise TypeError(f"{t!r} is not a readable type (GeoImage, GeoArray['y x'], …)")
    return spec


def spec_of(annotation: Any) -> ReadableType | None:
    """The `ReadableType` an annotation carries (``GeoImage``, ``X | None``)."""
    if typing.get_origin(annotation) is Annotated:
        for meta in annotation.__metadata__:
            if isinstance(meta, ReadableType):
                return meta
        return None
    if typing.get_origin(annotation) in (typing.Union, _types.UnionType):
        for arg in typing.get_args(annotation):
            spec = spec_of(arg)
            if spec is not None:
                return spec
    return None


def annotate(carrier: type, pattern: str, name: str | None = None) -> Any:
    """``Annotated[carrier, ReadableType]`` for ``carrier["band y x"]``."""
    kind = _CARRIER_KINDS[carrier.__name__]
    # built at run time, so the form is dynamic to a type checker
    annotated: Any = Annotated
    return annotated[carrier, ReadableType(name or pattern, pattern, kind)]


#: the kind each geoarray carrier class reads as (filled by the carriers' modules)
_CARRIER_KINDS: dict[str, str] = {
    "GeoArray": "grid",
    "GeoSwath": "swath",
    "GeoPoints": "points",
}


def _hint(fn_name: str, x: Any, t: ReadableType) -> str:
    """The fix for a carrier with extra leading dims: map over the first one."""
    kind = kind_of(x)
    if kind is None or kind.kind != t.kind:
        return ""
    sig = _canonical_signature(x, kind)
    if sig is None:
        return ""
    dims = tuple(d for d, _ in sig.dims)
    wanted = t.dims
    if "..." not in wanted and len(dims) == len(wanted) + 1 and dims[1:] == wanted:
        return f" Map over {dims[0]} with geoarray.map_over(x, {fn_name}, {dims[0]!r})."
    return ""


def check_dims[F: Callable[..., Any]](fn: F) -> F:
    """Check readable-type annotations at the call: arguments and the return.

    Arguments annotated with a readable type (``GeoImage``,
    ``GeoArray["band y x"]``, or either ``| None``) must match it
    (`matches`); so must the return value. Unannotated arguments pass.
    The checks read static dims only, so under ``jit`` they run once per
    trace and add nothing to the compiled program.

    Raises:
        DimsError: At the call, naming the argument, the type and what the
            carrier has, with the fix when an extra leading dim is the
            difference.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> from geoarray.types import GeoImage, GeoMap
        >>> @check_dims
        ... def first_band(scene: GeoImage) -> GeoMap:
        ...     return scene.isel(band=0)
        >>> stack = GeoArray(np.zeros((3, 2, 4, 4)), (1, 0, 0, 0, -1, 0), None,
        ...                  dims="time band y x")
        >>> first_band(stack)
        Traceback (most recent call last):
        ...
        geoarray._src.types.DimsError: first_band: scene is a GeoImage[grid: band y x], got dims ('time', 'band', 'y', 'x'). Map over time with geoarray.map_over(x, first_band, 'time').
    """  # noqa: E501
    hints = typing.get_type_hints(fn, include_extras=True)
    signature = inspect.signature(fn)
    specs = {
        name: spec
        for name in signature.parameters
        if name in hints and (spec := spec_of(hints[name])) is not None
    }
    returned = spec_of(hints["return"]) if "return" in hints else None
    name = getattr(fn, "__qualname__", repr(fn))

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        for arg, spec in specs.items():
            if arg not in bound.arguments or bound.arguments[arg] is None:
                continue
            value = bound.arguments[arg]
            why = _mismatch(value, spec)
            if why is not None:
                raise DimsError(
                    f"{name}: {arg} is a {spec!r}, got {why}."
                    + _hint(name, value, spec)
                )
        out = fn(*args, **kwargs)
        if returned is not None and out is not None:
            why = _mismatch(out, returned)
            if why is not None:
                raise DimsError(f"{name} returns a {returned!r}, but returned {why}.")
        return out

    return typing.cast(F, wrapper)
