"""Tests for ModelOp."""

from __future__ import annotations

import numpy as np
import pytest

from geotoolz.learn import ModelOp


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
        op = ModelOp(model=_Doubler())
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(op(arr), np.array([2.0, 4.0, 6.0]))


class TestMethodKwarg:
    def test_invokes_named_method(self) -> None:
        op = ModelOp(model=_SklearnLike(), method="predict")
        arr = np.array([10, 20, 30])
        np.testing.assert_array_equal(op(arr), np.array([11, 21, 31]))


class TestModelValidation:
    """The model's shape is checked at construction (#143)."""

    def test_predict_rejects_non_predictor(self) -> None:
        with pytest.raises(TypeError, match="Predictor"):
            ModelOp(model=_Doubler(), method="predict")

    def test_predict_accepts_predictor(self) -> None:
        from pipekit.protocols import Predictor

        model = _SklearnLike()
        assert isinstance(model, Predictor)
        assert ModelOp(model=model, method="predict").model is model

    def test_call_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            ModelOp(model=_SklearnLike())

    def test_named_method_must_exist(self) -> None:
        with pytest.raises(TypeError, match="'decision_function'"):
            ModelOp(model=_SklearnLike(), method="decision_function")

    def test_output_is_not_rewrapped(self) -> None:
        from _helpers import toy_geotensor
        from georeader.geotensor import GeoTensor

        out = ModelOp(model=_Doubler())(toy_geotensor(np.ones((2, 3, 3))))
        assert type(out) is np.ndarray
        assert not isinstance(out, GeoTensor)


class TestBatching:
    def test_batched_invocation_matches_non_batched(self) -> None:
        arr = np.arange(40).reshape(20, 2)
        non_batched = ModelOp(model=_Doubler())(arr)
        batched = ModelOp(model=_Doubler(), batch_size=4)(arr)
        np.testing.assert_array_equal(non_batched, batched)

    def test_batched_with_remainder_chunk(self) -> None:
        # 23 not evenly divisible by 5 — last chunk has 3 rows
        arr = np.arange(23).reshape(23, 1)
        result = ModelOp(model=_Doubler(), batch_size=5)(arr)
        np.testing.assert_array_equal(result, arr * 2)

    def test_batched_empty_input(self) -> None:
        # Empty input with batch_size set: must not raise on np.concatenate.
        # Pass the empty array straight to the model in one call.
        arr = np.zeros((0, 3), dtype=np.float32)
        result = ModelOp(model=_Doubler(), batch_size=8)(arr)
        assert result.shape == (0, 3)


class TestGetConfig:
    def test_config_records_model_type_and_method(self) -> None:
        op = ModelOp(model=_SklearnLike(), method="predict", batch_size=8)
        assert op.get_config() == {
            "model_type": "_SklearnLike",
            "method": "predict",
            "batch_size": 8,
        }


class TestFlags:
    def test_forbid_in_yaml(self) -> None:
        assert ModelOp.forbid_in_yaml is True
