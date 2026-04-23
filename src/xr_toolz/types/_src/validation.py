"""CF-metadata-aware validators for downloaded / in-memory data.

Validators are cheap checks that verify an :class:`xarray.DataArray` or
:class:`xarray.Dataset` matches a :class:`Variable` description:

- presence of the variable
- CF attribute agreement (``standard_name``, ``units``)
- value range inside :attr:`Variable.valid_range`
- dtype agreement with :attr:`Variable.dtype`

Failures surface as a :class:`ValidationReport` — a list of
:class:`Issue` records — rather than exceptions, so callers can decide
whether to warn, raise, or autofix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import xarray as xr

from xr_toolz.types._src.variable import Variable, resolve


class Severity(StrEnum):
    """Two-level severity for validation issues."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """Single validation finding."""

    variable: str
    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    """Aggregate validation result with convenience accessors."""

    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def raise_if_errors(self) -> None:
        """Raise :class:`ValueError` if any error-level issues are present."""
        errs = self.errors()
        if errs:
            lines = [f"[{i.code}] {i.variable}: {i.message}" for i in errs]
            raise ValueError("Validation failed:\n  " + "\n  ".join(lines))


def validate_variable(
    da: xr.DataArray,
    variable: str | Variable,
    *,
    check_range: bool = True,
    check_units: bool = True,
    check_standard_name: bool = True,
    check_dtype: bool = False,
) -> ValidationReport:
    """Validate a single :class:`xarray.DataArray` against a :class:`Variable`.

    Args:
        da: Array under test.
        variable: Either a :class:`Variable` or a name looked up in
            :data:`xr_toolz.types.REGISTRY`.
        check_range: If ``True`` and the variable defines ``valid_range``,
            flag finite values falling outside as errors.
        check_units: Compare ``da.attrs["units"]`` to the canonical
            units. Missing ``units`` is a warning; mismatch is an error.
        check_standard_name: Compare ``da.attrs["standard_name"]`` to
            the canonical standard name. Missing is a warning; mismatch
            is an error.
        check_dtype: Require ``da.dtype == variable.dtype`` (opt-in).

    Returns:
        A :class:`ValidationReport` aggregating all findings.
    """
    var = resolve(variable)
    report = ValidationReport()

    if check_standard_name and var.standard_name is not None:
        attr = da.attrs.get("standard_name")
        if attr is None:
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.WARNING,
                    code="missing_standard_name",
                    message=f"no standard_name; expected {var.standard_name!r}",
                )
            )
        elif attr != var.standard_name:
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.ERROR,
                    code="wrong_standard_name",
                    message=f"standard_name={attr!r}, expected {var.standard_name!r}",
                )
            )

    if check_units and var.units is not None:
        attr = da.attrs.get("units")
        if attr is None:
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.WARNING,
                    code="missing_units",
                    message=f"no units; expected {var.units!r}",
                )
            )
        elif attr != var.units:
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.ERROR,
                    code="wrong_units",
                    message=f"units={attr!r}, expected {var.units!r}",
                )
            )

    if check_range and var.valid_range is not None:
        lo, hi = var.valid_range
        finite_vals = da.where(np.isfinite(da))
        # Scalarize the count so the `if` doesn't try to bool-check a
        # DataArray (unambiguous on 0-d in practice, but brittle).
        n_finite = int(finite_vals.count().item())
        mn = float(finite_vals.min().values) if n_finite > 0 else None
        mx = float(finite_vals.max().values) if n_finite > 0 else None
        if mn is not None and (mn < lo or (mx is not None and mx > hi)):
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.ERROR,
                    code="out_of_range",
                    message=(
                        f"values in [{mn:.3g}, {mx:.3g}] fall outside "
                        f"valid_range [{lo}, {hi}]"
                    ),
                )
            )

    if check_dtype and var.dtype is not None and str(da.dtype) != var.dtype:
        report.add(
            Issue(
                variable=var.name,
                severity=Severity.ERROR,
                code="wrong_dtype",
                message=f"dtype={da.dtype}, expected {var.dtype}",
            )
        )

    return report


def validate_dataset(
    ds: xr.Dataset,
    variables: list[str | Variable],
    **kwargs: bool,
) -> ValidationReport:
    """Validate multiple variables in a dataset.

    Missing variables are reported as errors; present ones are checked
    via :func:`validate_variable`. Keyword arguments are forwarded.
    """
    report = ValidationReport()
    for spec in variables:
        var = resolve(spec)
        if var.name not in ds.data_vars:
            report.add(
                Issue(
                    variable=var.name,
                    severity=Severity.ERROR,
                    code="missing_variable",
                    message=(
                        f"{var.name!r} not in dataset; have {sorted(ds.data_vars)}"
                    ),
                )
            )
            continue
        sub = validate_variable(ds[var.name], var, **kwargs)
        report.issues.extend(sub.issues)
    return report


def apply_cf_attrs(
    da: xr.DataArray,
    variable: str | Variable,
    *,
    overwrite: bool = False,
) -> xr.DataArray:
    """Stamp CF attributes from ``variable`` onto ``da``.

    By default, existing attrs are kept. Set ``overwrite=True`` to
    replace them with the canonical values from the registry.
    """
    var = resolve(variable)
    # Shallow copy: we only mutate ``attrs`` below, so there's no reason
    # to duplicate the underlying array buffer.
    out = da.copy(deep=False)
    for k, v in var.cf_attrs().items():
        if overwrite or k not in out.attrs:
            out.attrs[k] = v
    return out
