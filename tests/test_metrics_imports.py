"""Import-surface tests for :mod:`xr_toolz.metrics` — guards against
regressions in the public API as the validation framework lands.
"""

from __future__ import annotations

import importlib
import warnings

import pytest


# ---- New canonical surface ------------------------------------------------


def test_metrics_package_root_exposes_layer0_and_operators():
    from xr_toolz.metrics import (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        PSDScore,
        R2Score,
        bias,
        correlation,
        find_intercept_1D,
        mae,
        mse,
        nrmse,
        psd_error,
        psd_score,
        r2_score,
        resolved_scale,
        rmse,
    )

    # Sanity-check at least one is callable.
    assert callable(rmse)
    assert RMSE.__name__ == "RMSE"
    assert PSDScore.__name__ == "PSDScore"
    # Hit unused-import suppression via tuple to keep the comprehensive list.
    _ = (
        MAE,
        MSE,
        NRMSE,
        Bias,
        Correlation,
        R2Score,
        bias,
        correlation,
        find_intercept_1D,
        mae,
        mse,
        nrmse,
        psd_error,
        psd_score,
        r2_score,
        resolved_scale,
    )


def test_metrics_pixel_submodule_imports():
    from xr_toolz.metrics.pixel import (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        R2Score,
        bias,
        correlation,
        mae,
        mse,
        nrmse,
        r2_score,
        rmse,
    )

    assert callable(mse)
    _ = (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        R2Score,
        bias,
        correlation,
        mae,
        nrmse,
        r2_score,
        rmse,
    )


def test_metrics_spectral_submodule_imports():
    from xr_toolz.metrics.spectral import (
        PSDScore,
        find_intercept_1D,
        psd_error,
        psd_score,
        resolved_scale,
    )

    assert callable(psd_score)
    _ = (PSDScore, find_intercept_1D, psd_error, resolved_scale)


def test_metrics_operators_submodule_imports():
    from xr_toolz.metrics.operators import (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        PSDScore,
        R2Score,
    )

    assert RMSE.__name__ == "RMSE"
    _ = (MAE, MSE, NRMSE, PSDScore, Bias, Correlation, R2Score)


@pytest.mark.parametrize(
    "submodule",
    [
        "forecast",
        "multiscale",
        "structural",
        "probabilistic",
        "distributional",
        "masked",
        "physical",
        "lagrangian",
    ],
)
def test_metrics_view_stub_submodules_are_importable(submodule):
    """Importing any V1–V5 stub submodule must succeed even before its
    epic lands its bodies — this is what unblocks the additive merge order.
    The stub must also expose no public names today; the V epic that
    fills it in is responsible for the public surface.

    ``object`` is excluded: F1.3 pre-reserves the canonical long-form
    V5 class names there. See ``test_metrics_object_reserved_names``.
    """
    mod = importlib.import_module(f"xr_toolz.metrics.{submodule}")
    public_names = [n for n in dir(mod) if not n.startswith("_")]
    assert public_names == [], (
        f"xr_toolz.metrics.{submodule} unexpectedly exports public names: "
        f"{public_names}"
    )


_RESERVED_OBJECT_CLASSES = (
    "ProbabilityOfDetection",
    "FalseAlarmRatio",
    "CriticalSuccessIndex",
    "IntersectionOverUnion",
    "DurationError",
    "IntensityBias",
    "CentroidDistance",
)


def test_metrics_object_exposes_all_reserved_names():
    """``metrics.object`` reserves the V5 long-form names per D14."""
    import xr_toolz.metrics.object as obj

    assert set(_RESERVED_OBJECT_CLASSES) <= set(dir(obj))


@pytest.mark.parametrize("name", _RESERVED_OBJECT_CLASSES)
def test_metrics_object_reserved_class_raises_notimplementederror(name):
    """Each V5-reserved class must raise ``NotImplementedError`` on
    instantiation, with a message that names the class and points at V5.
    """
    from xr_toolz.metrics import object as obj_module

    cls = getattr(obj_module, name)
    with pytest.raises(NotImplementedError) as exc_info:
        cls()
    msg = str(exc_info.value)
    assert name in msg, f"error message for {name} does not name the class: {msg!r}"
    assert "V5" in msg, f"error message for {name} does not point at V5: {msg!r}"


# ---- Legacy deprecation surface ------------------------------------------


def test_legacy_geo_metric_imports_warn_but_resolve():
    """``from xr_toolz.geo import rmse`` must keep working for one
    release with a :class:`DeprecationWarning`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from xr_toolz.geo import rmse  # noqa: F401

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning on legacy xr_toolz.geo.rmse"
    assert "xr_toolz.metrics" in str(deprecations[0].message)


def test_legacy_geo_operators_metric_imports_warn_but_resolve():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from xr_toolz.geo.operators import RMSE  # noqa: F401

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning on geo.operators.RMSE"
    assert "xr_toolz.metrics.operators" in str(deprecations[0].message)


def test_plain_import_xr_toolz_geo_is_silent():
    """Importing the package itself (without naming a moved metric) must
    not emit a :class:`DeprecationWarning` — the warning is per-name.
    """
    import xr_toolz.geo

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(xr_toolz.geo)

    metric_deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "xr_toolz.geo." in str(w.message)
        and "is deprecated" in str(w.message)
    ]
    assert not metric_deprecations, (
        f"plain import of xr_toolz.geo emitted unexpected deprecation warnings: "
        f"{[str(w.message) for w in metric_deprecations]}"
    )
