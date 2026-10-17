"""The two array backends: which one an array belongs to, and moving between them.

geoarray supports exactly two backends, NumPy and JAX. JAX is optional: no
function here imports it unless the caller already holds JAX data or asks
for it (`import_jax`), so a NumPy-only install never loads it.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Literal, Protocol

import numpy as np

from geoarray._src.errors import X64Error


__all__ = [
    "Backend",
    "SupportsArray",
    "backend_of",
    "import_jax",
    "is_jax_array",
    "is_tracer",
    "jax_loaded",
    "namespace",
    "require_x64",
]

Backend = Literal["numpy", "jax"]


class SupportsArray(Protocol):
    """What a GeoArray needs from its data: ``numpy.ndarray`` and ``jax.Array``."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    def astype(self, dtype: Any, /) -> Any: ...

    def __getitem__(self, key: Any, /) -> Any: ...


_JAX_HINT = "JAX data needs the jax extra: pip install 'geotoolz-array[jax]'"


def jax_loaded() -> bool:
    """Whether JAX is already imported in this process (never imports it)."""
    return "jax" in sys.modules


def import_jax() -> ModuleType:
    """Import JAX, or raise `ImportError` naming the ``[jax]`` extra."""
    try:
        import jax
    except ImportError as exc:  # pragma: no cover - exercised by the base-install job
        raise ImportError(_JAX_HINT) from exc
    return jax


def is_jax_array(x: Any) -> bool:
    """Whether ``x`` is a JAX array or tracer, without importing JAX."""
    if not jax_loaded():
        return False
    import jax

    return isinstance(x, jax.Array)


def is_tracer(x: Any) -> bool:
    """Whether ``x`` is a JAX tracer (inside ``jit``, ``grad``, ``vmap``, …)."""
    if not jax_loaded():
        return False
    import jax.core

    return isinstance(x, jax.core.Tracer)


def backend_of(x: Any) -> Backend:
    """The backend ``x`` belongs to: ``"numpy"`` or ``"jax"``.

    Args:
        x: A ``numpy.ndarray`` or a ``jax.Array`` (tracers included).

    Returns:
        ``"numpy"`` or ``"jax"``.

    Raises:
        TypeError: When ``x`` is neither.

    Example:
        >>> import numpy as np
        >>> backend_of(np.zeros((2, 3)))
        'numpy'
    """
    if isinstance(x, np.ndarray):
        return "numpy"
    if is_jax_array(x):
        return "jax"
    raise TypeError(
        f"GeoArray data must be a numpy.ndarray or a jax.Array, got {type(x).__name__}"
    )


def namespace(x: Any) -> ModuleType:
    """The array module for ``x``: ``numpy`` or ``jax.numpy``.

    Example:
        >>> import numpy as np
        >>> namespace(np.zeros(3)).__name__
        'numpy'
    """
    if backend_of(x) == "numpy":
        return np
    import jax.numpy as jnp

    return jnp


def require_x64() -> None:
    """Raise `X64Error` unless JAX runs with ``jax_enable_x64``."""
    jax = import_jax()
    if not jax.config.jax_enable_x64:
        raise X64Error(
            "a JAX GeoArray keeps its transform in float64, which needs 64-bit "
            "JAX: call jax.config.update('jax_enable_x64', True) at startup "
            "(data keeps its own dtype, e.g. float32)"
        )
