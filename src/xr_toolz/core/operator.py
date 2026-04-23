"""Base class for all xr_toolz operators.

Every operator is a callable with a uniform interface. Layer 1 operators
wrap pure functions with configuration and composition support; the
functional Graph API (Layer 2) reuses the same classes by detecting
symbolic ``Node`` arguments at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from xr_toolz.core.sequential import Sequential


class Operator:
    """Base class for all xr_toolz operators.

    Subclasses implement :meth:`_apply` with the real computation.
    :meth:`__call__` dispatches: if any positional argument is a
    :class:`~xr_toolz.core.graph.Node`, the call is interpreted as
    graph construction and returns a new ``Node``; otherwise it invokes
    :meth:`_apply` eagerly on the data.

    Subclasses should also override :meth:`get_config` so that operators
    are introspectable and JSON-serializable.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from xr_toolz.core.graph import Node

        if any(isinstance(a, Node) for a in args):
            return Node(operator=self, parents=args)
        return self._apply(*args, **kwargs)

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the operator eagerly on concrete data.

        Subclasses must override this. The base implementation raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `_apply`.")

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of constructor arguments.

        Combined with the class name, this is sufficient to reconstruct
        the operator. Rich state (arrays, fitted grids) should be passed
        as pre-computed objects by the caller and referenced by path in
        the config.
        """
        return {}

    def __repr__(self) -> str:
        config = self.get_config()
        params = ", ".join(f"{k}={v!r}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"

    def __or__(self, other: Operator) -> Sequential:
        """Pipe syntax: ``op_a | op_b`` builds ``Sequential([op_a, op_b])``."""
        from xr_toolz.core.sequential import Sequential

        if isinstance(other, Sequential):
            return Sequential([self, *other.operators])
        return Sequential([self, other])
