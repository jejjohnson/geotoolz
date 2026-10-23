"""KernelSpec and KernelOp: one L1 kernel on every carrier (pipekit#103).

``KernelOp(kernel, "(c)->()", core={"c": "band"})`` runs ``kernel`` on raw
arrays, on NumPy and JAX carriers (move core dims last, broadcast the rest
by name, call, rewrap) and on ``xr.DataArray`` (one ``apply_ufunc``).

Two rules the prototype had to add to ``pipekit_kernels.md``:

- **Outputs keep the first input's dim order.** A gufunc puts core dims
  last, so a per-column score ``(r,c)->(r)`` with ``core={"r": "y"}`` comes
  back as ``(x, y)``. It is transposed to the first input's order, so it is
  a ``GeoArray["y x"]`` again; dims new to the output go last.
- **A core dim that only an output has needs its size declared.** dask
  must know an output's size before running (``background_stats`` makes
  ``(c, d)`` from ``(r, c)``). ``KernelSpec(sizes={"d": "c"})`` says d has
  c's size.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable
from typing import Any

from proto.adapters import (
    BackendError,
    GeoArrayAdapter,
    XarrayAdapter,
    adapter_for,
    backend_name,
    merged_labels,
)


_GROUP = re.compile(r"\(([^)]*)\)")


def _parse(part: str) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(s.strip() for s in g.split(",") if s.strip())
        for g in _GROUP.findall(part)
    )


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """A generalized-ufunc signature over abstract core dims."""

    signature: str
    backends: tuple[str, ...] = ("numpy", "jax", "xarray")
    vectorize: bool = False
    #: output-only core dim → the input core dim whose size it has
    sizes: tuple[tuple[str, str], ...] = ()

    @property
    def inputs(self) -> tuple[tuple[str, ...], ...]:
        """Core dims of each input."""
        return _parse(self.signature.split("->")[0])

    @property
    def outputs(self) -> tuple[tuple[str, ...], ...]:
        """Core dims of each output."""
        return _parse(self.signature.split("->")[1])


class KernelOp:
    """An L1 kernel wrapped once, dispatched on the carriers' adapters."""

    def __init__(
        self,
        kernel: Callable[..., Any],
        spec: KernelSpec | str,
        core: dict[str, str],
        geometry: dict[str, Callable[[Any], Any]] | None = None,
        **params: Any,
    ) -> None:
        self.kernel = kernel
        self.spec = spec if isinstance(spec, KernelSpec) else KernelSpec(spec)
        self.core = dict(core)
        self.geometry = dict(geometry or {})
        self.params = params

    def __call__(self, *xs: Any) -> Any:
        if len(xs) != len(self.spec.inputs):
            raise TypeError(
                f"{self.spec.signature} takes {len(self.spec.inputs)} inputs"
            )
        adapters = [adapter_for(x) for x in xs]
        batch = getattr(xs[0], "batch_dims", ())
        if self.geometry and batch:
            # F4: derived geometry of a batched carrier varies along the batch
            # dims, and keyword inputs are passed whole, never mapped.
            raise ValueError(
                f"geometry= on a carrier batched over {batch}: "
                f"use map_over(op, x, {batch[0]!r}) so each item has one transform"
            )
        kwargs = self.params | {name: fn(xs[0]) for name, fn in self.geometry.items()}
        if all(a is None for a in adapters):
            return self._call_raw(xs, kwargs, backend_name(xs[0]))
        if any(isinstance(a, XarrayAdapter) for a in adapters):
            return self._call_xarray(xs, adapters, kwargs)
        return self._call_named(xs, adapters, kwargs)

    # -- raw arrays ---------------------------------------------------------

    def _func(self, kwargs: dict[str, Any], backend: str) -> Callable[..., Any]:
        f = functools.partial(self.kernel, **kwargs)
        if not self.spec.vectorize:
            return f
        sig = self.spec.signature.replace(" ", "")
        if backend == "jax":
            import jax.numpy as jnp

            return jnp.vectorize(f, signature=sig)
        import numpy as np

        return np.vectorize(f, signature=sig)

    def _call_raw(
        self, xs: tuple[Any, ...], kwargs: dict[str, Any], backend: str
    ) -> Any:
        return self._func(kwargs, backend)(*xs)

    # -- carriers on NumPy or JAX -------------------------------------------

    def _call_named(
        self, xs: tuple[Any, ...], adapters: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        for i, a in enumerate(adapters):
            if a is None:
                raise TypeError(
                    f"input {i} is a raw array next to named carriers; name it"
                )
        backends = {backend_name(a.data(x)) for a, x in zip(adapters, xs, strict=True)}
        if len(backends) > 1:
            raise BackendError(
                f"inputs on {sorted(backends)}: call to_jax() on the NumPy carriers"
            )
        backend = backends.pop()
        if backend not in self.spec.backends:
            raise BackendError(
                f"kernel declares backends {self.spec.backends}, not {backend}"
            )
        xs = _check_by_kind(xs, adapters)
        core_in = [tuple(self.core[c] for c in grp) for grp in self.spec.inputs]
        loop: list[str] = []
        sizes: dict[str, int] = {}
        for x, a, cores in zip(xs, adapters, core_in, strict=True):
            dims = a.dims(x)
            for c in cores:
                if c not in dims:
                    raise ValueError(f"core dim {c!r} not in {dims}")
            for d, n in zip(dims, a.data(x).shape, strict=True):
                if d not in cores and d not in loop:
                    loop.append(d)
                if d not in cores and sizes.setdefault(d, n) != n:
                    raise ValueError(f"dim {d!r} has sizes {sizes[d]} and {n}")
        xp = _xp(backend)
        arrays = [
            _arrange(xp, a.data(x), a.dims(x), loop, cores)
            for x, a, cores in zip(xs, adapters, core_in, strict=True)
        ]
        result = self._func(kwargs, backend)(*arrays)
        many = len(self.spec.outputs) > 1
        results = result if many else (result,)
        first_dims = adapters[0].dims(xs[0])
        template, tad = _template(xs, adapters)
        wrapped = []
        for out, grp in zip(results, self.spec.outputs, strict=True):
            dims = tuple(loop) + tuple(self.core[c] for c in grp)
            order = [d for d in first_dims if d in dims] + [
                d for d in dims if d not in first_dims
            ]
            out = xp.transpose(out, [dims.index(d) for d in order])
            order_t = tuple(order)
            wrapped.append(
                tad.rewrap(template, out, order_t, merged_labels(xs, order_t))
            )
        return tuple(wrapped) if many else wrapped[0]

    # -- xarray ---------------------------------------------------------------

    def _call_xarray(
        self, xs: tuple[Any, ...], adapters: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        import xarray as xr

        if not all(isinstance(a, XarrayAdapter) for a in adapters):
            raise BackendError(
                "mix of xarray and other carriers; convert with to_xarray()"
            )
        xs = adapters[0].check_compatible(*xs)
        in_dims = [[self.core[c] for c in grp] for grp in self.spec.inputs]
        out_dims = [[self.core[c] for c in grp] for grp in self.spec.outputs]
        known = {}
        for x, grp in zip(xs, self.spec.inputs, strict=True):
            for c in grp:
                known[c] = x.sizes[self.core[c]]
        out_sizes = {self.core[o]: known[i] for o, i in self.spec.sizes}
        dtype = xs[0].dtype
        result = xr.apply_ufunc(
            functools.partial(self.kernel, **kwargs),
            *xs,
            input_core_dims=in_dims,
            output_core_dims=out_dims,
            vectorize=self.spec.vectorize,
            dask="parallelized",
            join="exact",
            output_dtypes=[dtype] * len(out_dims),
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": out_sizes},
        )
        first_dims = tuple(xs[0].dims)

        def order(da: Any) -> Any:
            dims = tuple(da.dims)
            o = [d for d in first_dims if d in dims] + [
                d for d in dims if d not in first_dims
            ]
            return da.transpose(*o)

        if isinstance(result, tuple):
            return tuple(order(r) for r in result)
        return order(result)


def _xp(backend: str) -> Any:
    if backend == "jax":
        import jax.numpy as jnp

        return jnp
    import numpy as np

    return np


def _arrange(
    xp: Any, data: Any, dims: tuple[str, ...], loop: list[str], cores: tuple[str, ...]
) -> Any:
    """Transpose to (present loop dims, in loop order) + cores; size 1 for the rest."""
    present = [d for d in loop if d in dims]
    moved = xp.transpose(data, [dims.index(d) for d in present + list(cores)])
    index = tuple(slice(None) if d in dims else None for d in loop) + (
        slice(None),
    ) * len(cores)
    return moved[index]


def _check_by_kind(xs: tuple[Any, ...], adapters: list[Any]) -> tuple[Any, ...]:
    """Grid checks between carriers of the same kind; size checks across kinds."""
    out = list(xs)
    groups: dict[int, list[int]] = {}
    for i, a in enumerate(adapters):
        groups.setdefault(id(a), []).append(i)
    for idx in groups.values():
        a = adapters[idx[0]]
        checked = a.check_compatible(*[xs[i] for i in idx])
        for i, x in zip(idx, checked, strict=True):
            out[i] = x
    return tuple(out)


def _template(xs: tuple[Any, ...], adapters: list[Any]) -> tuple[Any, Any]:
    """The input outputs are rewrapped on: the first GeoArray, else the first."""
    for x, a in zip(xs, adapters, strict=True):
        if isinstance(a, GeoArrayAdapter):
            return x, a
    return xs[0], adapters[0]
