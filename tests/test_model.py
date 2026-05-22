"""Tests for ModelOp."""

from __future__ import annotations

import numpy as np

from geotoolz.model import ModelOp


class _Doubler:
    """Plain-callable model that doubles its input."""

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return arr * 2


class _SklearnLike:
    """Stand-in for sklearn — only ``predict`` works, not ``__call__``."""

    def predict(self, arr: np.ndarray) -> np.ndarray:
        return arr + 1


class TestCallableModel:
    def test_invokes_call(self) -> None:
        op = ModelOp(_Doubler())
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(op(arr), np.array([2.0, 4.0, 6.0]))


class TestMethodKwarg:
    def test_invokes_named_method(self) -> None:
        op = ModelOp(_SklearnLike(), method="predict")
        arr = np.array([10, 20, 30])
        np.testing.assert_array_equal(op(arr), np.array([11, 21, 31]))


class TestBatching:
    def test_batched_invocation_matches_non_batched(self) -> None:
        arr = np.arange(40).reshape(20, 2)
        non_batched = ModelOp(_Doubler())(arr)
        batched = ModelOp(_Doubler(), batch_size=4)(arr)
        np.testing.assert_array_equal(non_batched, batched)

    def test_batched_with_remainder_chunk(self) -> None:
        # 23 not evenly divisible by 5 — last chunk has 3 rows
        arr = np.arange(23).reshape(23, 1)
        result = ModelOp(_Doubler(), batch_size=5)(arr)
        np.testing.assert_array_equal(result, arr * 2)

    def test_batched_empty_input(self) -> None:
        # Empty input with batch_size set: must not raise on np.concatenate.
        # Pass the empty array straight to the model in one call.
        arr = np.zeros((0, 3), dtype=np.float32)
        result = ModelOp(_Doubler(), batch_size=8)(arr)
        assert result.shape == (0, 3)


class TestGetConfig:
    def test_config_records_model_type_and_method(self) -> None:
        op = ModelOp(_SklearnLike(), method="predict", batch_size=8)
        assert op.get_config() == {
            "model_type": "_SklearnLike",
            "method": "predict",
            "batch_size": 8,
        }


class TestFlags:
    def test_forbid_in_yaml(self) -> None:
        assert ModelOp.forbid_in_yaml is True
