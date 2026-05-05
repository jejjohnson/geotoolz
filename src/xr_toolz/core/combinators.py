"""Operator combinators — domain-agnostic wrappers over inner operators.

Three small utilities that come up repeatedly when building diagnostic
pipelines:

- :class:`Augment` — run an inner operator and *merge* its output back
  into the input Dataset, preserving the input's existing variables.
- :class:`Tap` — call a side-effect (logging, plotting, write-to-disk)
  on the input and pass the input through unchanged.
- :class:`ApplyToEach` — re-instantiate a prototype operator once per
  value of a chosen kwarg and merge all results together.

Each combinator is itself an :class:`Operator`, so it composes inside
``Sequential`` chains and ``Graph`` DAGs.

**Serialization caveat.** ``Augment`` and ``ApplyToEach`` carry nested
``Operator`` state. Their :meth:`Operator.get_config` outputs are
JSON-safe (no live instances inside) for *introspection* — printing,
logging, diffing pipeline structure — but the configs are **not**
constructor-replayable: a literal ``Augment(**cfg)`` /
``ApplyToEach(**cfg)`` round-trip fails, because the constructor
expects live ``Operator`` instances rather than serialized
``{"class", "config"}`` records. A future deserializer with a class
registry would close that gap; until then, the configs are
introspection-only.

The combinators address a structural mismatch between the Layer 1
contract — a single-input operator returns a Dataset, *replacing* the
input — and common pipeline use cases that want to *grow* a Dataset by
appending derived columns. ``Augment`` does the merge-back; ``Tap``
adds observability without altering the data; ``ApplyToEach`` fans out
a single prototype across multiple variables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import xarray as xr

from xr_toolz.core.operator import Operator


class Augment(Operator):
    """Run an inner ``Operator`` and merge its output back into the input.

    The inner operator must accept a Dataset and return a Dataset. The
    return value of :class:`Augment` is ``input.merge(inner(input))`` —
    the input's existing variables are preserved, and the inner op's
    new variables are added alongside.

    The ``merge`` is invoked with ``compat="no_conflicts"`` (xarray's
    safe default), so if the inner op produces a variable name that
    already exists in the input *and the values differ*, the merge
    raises rather than silently overwriting. Diagnostics that share an
    output name with one of their inputs (rare in practice) need to be
    renamed before being wrapped.

    The combinator is the canonical idiom for chaining diagnostics
    inside a ``Sequential``: each step augments the threaded Dataset
    with new columns, so later diagnostics can depend on variables
    added by earlier ones.

    Args:
        inner: The inner :class:`Operator`. Must accept a single
            ``xr.Dataset`` and return an ``xr.Dataset``. Standard
            single-input ``xr_toolz.ocn`` diagnostics
            (``RelativeVorticity``, ``KineticEnergy``,
            ``GeostrophicVelocities``, ``Divergence``, ...) satisfy
            this contract. Two-input metrics (``RMSE``, ``MAE``, ...)
            and operators that return a ``DataArray`` rather than a
            ``Dataset`` do **not** fit and will raise at call time.

    Raises:
        TypeError: If ``inner`` is not an :class:`Operator` (raised at
            construction), or if ``inner(ds)`` does not return a
            ``Dataset`` (raised at call time).
        xarray.MergeError: If the inner op's output collides with an
            existing input variable whose values do not match.

    Example:
        Build a single Dataset carrying every kinematic diagnostic
        for a velocity field::

            from xr_toolz import Augment, Sequential
            from xr_toolz.ocn.operators import (
                RelativeVorticity, KineticEnergy, OkuboWeiss,
            )

            diagnostics = Sequential([
                Augment(RelativeVorticity()),    # adds vort_r
                Augment(KineticEnergy()),         # adds ke
                Augment(OkuboWeiss()),            # adds ow
            ])
            enriched = diagnostics(velocity_dataset)
            # enriched has u, v, vort_r, ke, ow

        Diagnostics that depend on previously-added columns work
        transparently because ``Sequential`` threads the merged
        Dataset forward::

            Sequential([
                Augment(RelativeVorticity()),                # adds vort_r
                Augment(Enstrophy(variable="vort_r")),       # reads vort_r
            ])

    See Also:
        :class:`Sequential` — chain :class:`Augment` calls together.
        :class:`ApplyToEach` — augment with one diagnostic across many
            variables.
    """

    def __init__(self, inner: Operator) -> None:
        if not isinstance(inner, Operator):
            raise TypeError(
                f"Augment expects an Operator instance, got {type(inner).__name__}."
            )
        self.inner = inner

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        result = self.inner(ds)
        if not isinstance(result, xr.Dataset):
            raise TypeError(
                f"Augment requires the inner op to return a Dataset; "
                f"{self.inner.__class__.__name__} returned "
                f"{type(result).__name__}."
            )
        return ds.merge(result, compat="no_conflicts")

    def get_config(self) -> dict[str, Any]:
        return {
            "inner": {
                "class": self.inner.__class__.__name__,
                "config": self.inner.get_config(),
            }
        }

    def __repr__(self) -> str:
        return f"Augment({self.inner!r})"


class Tap(Operator):
    """Call a side-effect on the input, then return the input unchanged.

    ``side_effect`` is invoked as ``side_effect(ds)`` for its side
    effect only — its return value is discarded. The Dataset flows
    through unchanged, so ``Tap`` is safe to drop into a
    ``Sequential`` chain anywhere observability is wanted: progress
    logging, on-the-fly QC plotting, writing intermediate state to
    disk, or asserting invariants.

    The side-effect callable is **not** captured in
    :meth:`get_config` — arbitrary Python callables are not in general
    JSON-serializable. ``Tap`` therefore advertises only its
    ``name`` (defaulting to ``side_effect.__name__`` when available)
    and a literal ``"<callable>"`` placeholder. This means a pipeline
    that contains a ``Tap`` cannot be losslessly round-tripped through
    JSON; use :class:`Tap` for ergonomics, not for serialization.

    Args:
        side_effect: A callable taking the input ``xr.Dataset``. Its
            return value is ignored. May raise to abort the pipeline,
            but typically should not — the convention is "look but
            don't touch".
        name: Optional human-readable label for the tap. Defaults to
            ``side_effect.__name__`` if available, else
            ``"<callable>"``. Surfaces in :meth:`get_config` and the
            string repr.

    Raises:
        TypeError: If ``side_effect`` is not callable.

    Example:
        Log the size of the dataset at each stage of a pipeline::

            from xr_toolz import Tap, Sequential
            from xr_toolz.geo.operators import ValidateCoords, RemoveMean
            from xr_toolz.interpolate.operators import RegridLike

            def log_sizes(ds):
                print(f"  sizes: {dict(ds.sizes)}")

            pipe = Sequential([
                Tap(log_sizes, name="before validate"),
                ValidateCoords(),
                Tap(log_sizes, name="after validate"),
                RegridLike(target_grid),
                Tap(log_sizes, name="after regrid"),
                RemoveMean(("lat", "lon")),
            ])
            pipe(ds)
            # Prints sizes at three points without altering the pipeline.

        Save intermediate state to disk::

            def save_to_cache(ds):
                ds.to_netcdf(".cache/intermediate.nc")

            pipe = Sequential([
                preprocess,
                Tap(save_to_cache),    # write before the expensive step
                expensive_inference,
            ])

    See Also:
        :class:`Augment` — for ops that *do* modify the threaded
            dataset.
    """

    def __init__(
        self,
        side_effect: Callable[[xr.Dataset], Any],
        *,
        name: str | None = None,
    ) -> None:
        if not callable(side_effect):
            raise TypeError(
                f"Tap expects a callable side_effect, got {type(side_effect).__name__}."
            )
        self.side_effect = side_effect
        self.name = name or getattr(side_effect, "__name__", "<callable>")

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        self.side_effect(ds)
        return ds

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "side_effect": "<callable>"}

    def __repr__(self) -> str:
        return f"Tap(name={self.name!r})"


class ApplyToEach(Operator):
    """Re-instantiate a prototype operator per value and merge results.

    Given a prototype :class:`Operator`, a kwarg name, and a sequence
    of values, :class:`ApplyToEach` builds and runs one new instance
    of the prototype's class per value, varying the chosen kwarg.
    Each resulting Dataset is merged into the input with the same
    safe ``compat="no_conflicts"`` semantics as :class:`Augment`.

    The behaviour is equivalent to::

        Sequential([
            Augment(prototype.__class__(**{**proto_config, kwarg: v}))
            for v in values
        ])

    Each rebuilt operator runs against the **threaded** result, not
    the original input — i.e. the second pass sees the variables added
    by the first pass. This is what makes the equivalence with
    ``Sequential([Augment(...) for v in values])`` exact, and lets
    later sweeps consume variables produced by earlier ones.

    The prototype's ``get_config()`` is read once and used as the
    base kwargs dict for the rebuilt instances. The prototype's class
    must therefore satisfy the *leaf* contract
    ``cls(**proto.get_config())`` — i.e. its ``get_config`` must
    return a dict directly usable as constructor kwargs. All
    single-operator Layer 1 classes in ``xr_toolz`` satisfy this; the
    composite/combinator classes (``Sequential``, ``Graph``,
    ``Augment``, ``Tap``, ``ApplyToEach`` itself) do **not**, because
    their configs carry serialized child records rather than live
    operator instances. ``ApplyToEach`` validates this contract at
    construction time via a sentinel rebuild and raises
    ``TypeError`` with a descriptive message if it fails.

    Each rebuilt operator must produce uniquely-named output
    variables across the value sweep, otherwise the eventual merge
    raises. In practice most diagnostics name their output after the
    input variable they consume (e.g. ``Frontogenesis(scalar="ssh")``
    → ``ssh_frontogenesis``), so this is not usually a concern.

    Args:
        prototype: A configured single-operator :class:`Operator`
            instance. Composite classes such as ``Sequential`` or
            other combinators are not accepted (see contract above).
        kwarg: Name of the constructor kwarg to vary across the
            sweep. Must be a key of ``prototype.get_config()``.
        values: Sequence of values to substitute for ``kwarg``. Each
            entry is used to build one rebuilt operator.

    Raises:
        TypeError: If ``prototype`` is not an :class:`Operator`, or if
            ``prototype.get_config()`` cannot be passed back to its
            class constructor (composite-config case), or if any
            rebuilt op does not return a Dataset.
        ValueError: If ``kwarg`` is not present in
            ``prototype.get_config()``.

    Example:
        Compute frontogenesis for three scalars in one call::

            from xr_toolz import ApplyToEach
            from xr_toolz.ocn.operators import Frontogenesis

            multi_fg = ApplyToEach(
                Frontogenesis(scalar="ssh"),
                kwarg="scalar",
                values=["ssh", "thetao", "so"],
            )
            out = multi_fg(ds)
            # out has ssh_frontogenesis, thetao_frontogenesis,
            # so_frontogenesis merged in alongside the original
            # variables.

        Inside a ``Sequential`` that augments the dataset before and
        after::

            Sequential([
                ValidateCoords(),
                ApplyToEach(
                    Frontogenesis(scalar="ssh"),
                    kwarg="scalar",
                    values=["thetao", "so"],
                ),
                Augment(KineticEnergy()),
            ])

    See Also:
        :class:`Augment` — wrap a single inner op to merge its output.
    """

    def __init__(
        self,
        prototype: Operator,
        *,
        kwarg: str,
        values: Sequence[Any],
    ) -> None:
        if not isinstance(prototype, Operator):
            raise TypeError(
                f"ApplyToEach expects an Operator prototype, "
                f"got {type(prototype).__name__}."
            )
        base_config = prototype.get_config()
        if kwarg not in base_config:
            raise ValueError(
                f"ApplyToEach: kwarg {kwarg!r} not present in "
                f"{prototype.__class__.__name__}.get_config(); "
                f"available keys: {sorted(base_config)}."
            )
        # Fail fast if the prototype's config carries any serialized
        # child Operator records (the ``{"class", "config"}`` shape
        # used by Augment/ApplyToEach/Sequential/Graph for nested
        # operator state). Such a prototype cannot be replayed by
        # ``cls(**get_config())`` because the constructor expects
        # live Operator instances rather than dicts.
        cls = type(prototype)
        if _has_serialized_operator_record(base_config):
            raise TypeError(
                f"ApplyToEach: prototype {cls.__name__} cannot be "
                f"reconstructed from its own get_config() — its config "
                f"carries serialized child operator records "
                f'(``{{"class", "config"}}`` dicts) rather than live '
                f"Operator instances. This is a known limitation for "
                f"composite operators (Sequential, Graph, Augment, "
                f"ApplyToEach). Pass a single-operator prototype "
                f"whose get_config() returns plain constructor kwargs."
            )
        self.prototype = prototype
        self.kwarg = kwarg
        self.values = list(values)

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        cls = type(self.prototype)
        base = self.prototype.get_config()
        result = ds
        for value in self.values:
            inner = cls(**{**base, self.kwarg: value})
            # Run against the *threaded* result so later values can
            # depend on variables produced by earlier ones — matches
            # the documented Sequential([Augment(...)]) equivalence.
            out = inner(result)
            if not isinstance(out, xr.Dataset):
                raise TypeError(
                    f"ApplyToEach requires inner op to return a Dataset; "
                    f"{cls.__name__}(..., {self.kwarg}={value!r}) returned "
                    f"{type(out).__name__}."
                )
            result = result.merge(out, compat="no_conflicts")
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "prototype": {
                "class": self.prototype.__class__.__name__,
                "config": self.prototype.get_config(),
            },
            "kwarg": self.kwarg,
            "values": list(self.values),
        }

    def __repr__(self) -> str:
        return (
            f"ApplyToEach({self.prototype!r}, kwarg={self.kwarg!r}, "
            f"values={self.values!r})"
        )


def _has_serialized_operator_record(value: Any) -> bool:
    """Recurse into a config dict looking for ``{"class", "config"}`` markers.

    Used by :class:`ApplyToEach` to detect composite prototypes whose
    ``get_config()`` carries serialized child Operator records.
    """
    if isinstance(value, dict):
        if {"class", "config"} <= value.keys():
            return True
        return any(_has_serialized_operator_record(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_serialized_operator_record(v) for v in value)
    return False


__all__ = ["ApplyToEach", "Augment", "Tap"]
