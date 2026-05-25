"""Unit tests for `BlendMatched` — per-pixel fusion across N tensors.

`BlendMatched` is the per-pixel averaging counterpart to
`StackMatched` (which concatenates along bands). Three methods:

* ``"mean"`` — equal-weight average.
* ``"weighted_mean"`` — per-source scalar weights.
* ``"ivw"`` — inverse-variance weighting from per-source variance maps.

NaN-policy choices: ``"ignore"`` skips NaN samples (surviving
weights renormalise); ``"propagate"`` poisons the output pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from pipekit import Operator

from geotoolz.compositing import BlendMatched


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(values: np.ndarray, *, transform=None, crs="EPSG:32629") -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=transform
        or rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs=crs,
        fill_value_default=np.nan,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestBlendMatchedConstruction:
    def test_is_operator_with_config(self) -> None:
        op = BlendMatched(
            method="weighted_mean", weights=[1.0, 2.0], nan_policy="propagate"
        )
        assert isinstance(op, Operator)
        assert op.get_config() == {
            "method": "weighted_mean",
            "weights": [1.0, 2.0],
            "nan_policy": "propagate",
        }

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"method must be"):
            BlendMatched(method="median")  # type: ignore[arg-type]

    def test_invalid_nan_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"nan_policy"):
            BlendMatched(nan_policy="bubble")  # type: ignore[arg-type]

    def test_weighted_mean_requires_weights(self) -> None:
        with pytest.raises(ValueError, match=r"requires `weights`"):
            BlendMatched(method="weighted_mean")

    def test_weights_only_valid_for_weighted_mean(self) -> None:
        # Catching the mismatch upfront — silently ignoring `weights`
        # under method='mean' would be a confusing footgun.
        with pytest.raises(ValueError, match=r"only applies to"):
            BlendMatched(method="mean", weights=[1.0, 1.0])
        with pytest.raises(ValueError, match=r"only applies to"):
            BlendMatched(method="ivw", weights=[1.0, 1.0])


# ---------------------------------------------------------------------------
# `method="mean"` — equal-weight average
# ---------------------------------------------------------------------------


class TestMean:
    def test_two_2d_tensors(self) -> None:
        a = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 8.0, dtype=np.float32))
        result = BlendMatched()([a, b])
        arr = np.asarray(result)
        # Mean of 2 and 8 = 5.
        np.testing.assert_array_equal(arr, 5.0)

    def test_three_3d_tensors_preserve_band_axis(self) -> None:
        a = _gt(np.zeros((3, 4, 4), dtype=np.float32))
        b = _gt(np.full((3, 4, 4), 3.0, dtype=np.float32))
        c = _gt(np.full((3, 4, 4), 6.0, dtype=np.float32))
        result = BlendMatched()([a, b, c])
        arr = np.asarray(result)
        assert arr.shape == (3, 4, 4)
        # Mean(0, 3, 6) = 3.
        np.testing.assert_allclose(arr, 3.0, rtol=1e-9)

    def test_mapping_input(self) -> None:
        # BlendMatched accepts dict inputs (like StackMatched), reusing
        # the same `_normalize_to_sequence` helper. Order is dict-
        # insertion order; equal-weight mean is order-invariant.
        a = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 8.0, dtype=np.float32))
        result = BlendMatched()({"modis": a, "s2": b})
        np.testing.assert_array_equal(np.asarray(result), 5.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"at least one"):
            BlendMatched()([])

    def test_shape_mismatch_rejected(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        b = _gt(np.zeros((8, 8), dtype=np.float32))
        with pytest.raises(ValueError, match=r"share spatial shape"):
            BlendMatched()([a, b])

    def test_band_count_mismatch_rejected(self) -> None:
        # Per-pixel average across mismatched band counts is undefined.
        # (Use StackMatched if you want to concatenate.)
        a = _gt(np.zeros((2, 4, 4), dtype=np.float32))
        b = _gt(np.zeros((3, 4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r"full shape"):
            BlendMatched()([a, b])


# ---------------------------------------------------------------------------
# `method="weighted_mean"`
# ---------------------------------------------------------------------------


class TestWeightedMean:
    def test_scalar_weights_shift_mean(self) -> None:
        a = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 8.0, dtype=np.float32))
        # 1:3 weight → (1*2 + 3*8) / 4 = 26 / 4 = 6.5.
        result = BlendMatched(method="weighted_mean", weights=[1.0, 3.0])([a, b])
        np.testing.assert_allclose(np.asarray(result), 6.5, rtol=1e-9)

    def test_equal_weights_match_mean(self) -> None:
        a = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 8.0, dtype=np.float32))
        wm = BlendMatched(method="weighted_mean", weights=[1.0, 1.0])([a, b])
        m = BlendMatched()([a, b])
        np.testing.assert_allclose(np.asarray(wm), np.asarray(m), rtol=1e-9)

    def test_weight_count_mismatch_rejected(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        b = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r"weights"):
            BlendMatched(method="weighted_mean", weights=[1.0, 2.0, 3.0])([a, b])

    def test_works_on_3d_tensors(self) -> None:
        a = _gt(np.full((2, 4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((2, 4, 4), 5.0, dtype=np.float32))
        # 1:1 → mean 3; broadcast across (band, H, W).
        result = BlendMatched(method="weighted_mean", weights=[1.0, 1.0])([a, b])
        assert np.asarray(result).shape == (2, 4, 4)
        np.testing.assert_allclose(np.asarray(result), 3.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# `method="ivw"` — inverse-variance weighting
# ---------------------------------------------------------------------------


class TestInverseVarianceWeighting:
    def test_low_variance_source_dominates(self) -> None:
        # Source A: value 10, variance 1 → weight 1.0.
        # Source B: value 20, variance 100 → weight 0.01.
        # Combined: (1.0 * 10 + 0.01 * 20) / (1.0 + 0.01) = 10.099...
        a = _gt(np.full((4, 4), 10.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 20.0, dtype=np.float32))
        var_a = np.ones((4, 4), dtype=np.float32)
        var_b = np.full((4, 4), 100.0, dtype=np.float32)
        result = BlendMatched(method="ivw")([a, b], variances=[var_a, var_b])
        np.testing.assert_allclose(
            np.asarray(result), (10 / 1 + 20 / 100) / (1 / 1 + 1 / 100), rtol=1e-9
        )

    def test_equal_variances_match_mean(self) -> None:
        a = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 8.0, dtype=np.float32))
        var_eq = np.full((4, 4), 4.0, dtype=np.float32)
        result = BlendMatched(method="ivw")([a, b], variances=[var_eq, var_eq])
        np.testing.assert_allclose(np.asarray(result), 5.0, rtol=1e-9)

    def test_per_pixel_varying_variance(self) -> None:
        # Variance maps that vary across pixels — equivalent to weighing
        # each pixel differently per source.
        a = _gt(np.full((2, 2), 10.0, dtype=np.float32))
        b = _gt(np.full((2, 2), 20.0, dtype=np.float32))
        # A reliable in row 0, B reliable in row 1.
        var_a = np.array([[1.0, 1.0], [100.0, 100.0]], dtype=np.float32)
        var_b = np.array([[100.0, 100.0], [1.0, 1.0]], dtype=np.float32)
        result = BlendMatched(method="ivw")([a, b], variances=[var_a, var_b])
        arr = np.asarray(result)
        # Row 0 → A dominates → ~10.
        assert arr[0, 0] < 11.0 and arr[0, 1] < 11.0
        # Row 1 → B dominates → ~20.
        assert arr[1, 0] > 19.0 and arr[1, 1] > 19.0

    def test_ivw_requires_variances(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r"requires `variances`"):
            BlendMatched(method="ivw")([a])

    def test_variance_count_mismatch_rejected(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        b = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r"variance arrays"):
            BlendMatched(method="ivw")([a, b], variances=[np.ones((4, 4))])


# ---------------------------------------------------------------------------
# NaN policy
# ---------------------------------------------------------------------------


class TestNanPolicy:
    def test_ignore_drops_nan_from_blend(self) -> None:
        a_vals = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        b_vals = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = BlendMatched(nan_policy="ignore")([_gt(a_vals), _gt(b_vals)])
        arr = np.asarray(result)
        # (0, 0): mean(1, 5) = 3.
        assert arr[0, 0] == pytest.approx(3.0)
        # (0, 1): only B contributes → 6.
        assert arr[0, 1] == pytest.approx(6.0)
        # (1, 0): only B contributes → 7.
        assert arr[1, 0] == pytest.approx(7.0)
        # (1, 1): mean(4, 8) = 6.
        assert arr[1, 1] == pytest.approx(6.0)

    def test_ignore_all_nan_yields_nan(self) -> None:
        a = _gt(np.full((2, 2), np.nan, dtype=np.float32))
        b = _gt(np.full((2, 2), np.nan, dtype=np.float32))
        result = BlendMatched(nan_policy="ignore")([a, b])
        assert np.all(np.isnan(np.asarray(result)))

    def test_propagate_poisons_pixel(self) -> None:
        a_vals = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        b_vals = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = BlendMatched(nan_policy="propagate")([_gt(a_vals), _gt(b_vals)])
        arr = np.asarray(result)
        # (0, 0): mean(1, 5) = 3 — both valid.
        assert arr[0, 0] == pytest.approx(3.0)
        # (0, 1): NaN in A → output NaN under propagate.
        assert np.isnan(arr[0, 1])
        # The rest are valid.
        assert arr[1, 0] == pytest.approx(5.0)
        assert arr[1, 1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Geo metadata
# ---------------------------------------------------------------------------


class TestGeoMetadata:
    def test_output_carries_first_inputs_grid(self) -> None:
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 3.0, dtype=np.float32))
        result = BlendMatched()([a, b])
        assert result.transform == a.transform
        assert result.crs == a.crs

    def test_grid_mismatch_rejected(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        b = _gt(
            np.zeros((4, 4), dtype=np.float32),
            transform=rasterio.Affine(10.0, 0.0, 600_000.0, 0.0, -10.0, 4_000_000.0),
        )
        with pytest.raises(ValueError, match=r"share spatial shape, transform"):
            BlendMatched()([a, b])
