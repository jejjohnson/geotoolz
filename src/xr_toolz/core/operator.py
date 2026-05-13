"""Base class for all xr_toolz operators.

Every operator is a callable with a uniform interface. Layer 1 operators
wrap pure functions with configuration and composition support; the
functional Graph API (Layer 2) reuses the same classes by detecting
symbolic ``Node`` arguments at call time.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from xr_toolz.core.sequential import Sequential


class ConfigMixin:
    """Mixin that auto-implements ``get_config`` from constructor arguments.

    The wrapped ``__init__`` first runs the user's body (so attribute
    coercions take effect) and then captures one entry per parameter:
    if ``self.<name>`` was set during ``__init__``, the *normalized*
    value is captured; otherwise the raw bound argument is kept. This
    preserves the contract that ``get_config`` should be JSON-
    serializable as long as ``__init__`` casts numpy / pandas scalars
    to built-in types before storing them.

    Classes that need custom serialization (e.g. ndarray-valued
    configuration that requires a list round-trip) should override
    ``get_config`` directly or set ``__config_mixin_auto__ = False``.
    Per-field exclusion is available via ``__config_exclude__``.
    """

    __config_mixin_auto__ = True
    __config_exclude__: tuple[str, ...] = ()
    _config_mixin_config: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__config_mixin_auto__", True):
            return
        if "get_config" in cls.__dict__ or "__init__" not in cls.__dict__:
            return

        init = cls.__dict__["__init__"]
        signature = inspect.signature(init)

        def wrapped_init(self: ConfigMixin, *args: Any, **kwargs: Any) -> None:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            call_args: list[Any] = [self]
            call_kwargs: dict[str, Any] = {}
            for name, param in tuple(signature.parameters.items())[1:]:
                value = bound.arguments[name]
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    call_args.extend(value)
                elif param.kind is inspect.Parameter.VAR_KEYWORD:
                    call_kwargs.update(value)
                elif param.kind is inspect.Parameter.POSITIONAL_ONLY:
                    call_args.append(value)
                else:
                    call_kwargs[name] = value
            init(*call_args, **call_kwargs)
            exclude = set(getattr(self, "__config_exclude__", ()))
            # Prefer self.<name> when __init__ stored a coerced value (e.g.
            # `self.dj = float(dj)`); fall back to the raw bound argument.
            captured: dict[str, Any] = {}
            for name, value in bound.arguments.items():
                if name == "self" or name in exclude:
                    continue
                captured[name] = getattr(self, name, value)
            self._config_mixin_config = captured

        wrapped_init.__name__ = init.__name__
        wrapped_init.__qualname__ = init.__qualname__
        wrapped_init.__doc__ = init.__doc__
        wrapped_init.__signature__ = signature  # ty: ignore[unresolved-attribute]
        cls.__init__ = wrapped_init  # ty: ignore[invalid-assignment]

    def get_config(self) -> dict[str, Any]:
        """Return captured constructor arguments.

        Should be JSON-serializable. ``__init__`` is responsible for
        coercing scalar inputs to JSON-friendly types (``float(...)``,
        ``int(...)``, etc.) before assigning them to ``self``.
        """
        return dict(getattr(self, "_config_mixin_config", {}))


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

    def compute_output_signature(self, input_signature: Any) -> Any:
        """Infer the output signature without executing the operator.

        Shape-preserving operators inherit this default. Operators that
        rename, remove, or resize dimensions override it.
        """
        if isinstance(input_signature, tuple):
            if len(input_signature) != 1:
                raise ValueError(
                    f"{self.__class__.__name__} received {len(input_signature)} "
                    "input signatures but expected 1; override "
                    "compute_output_signature to handle multiple inputs."
                )
            return input_signature[0]
        return input_signature

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
