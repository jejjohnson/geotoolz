"""`get_config()` smoke tests for the matched patcher family.

Covers `MatchedSpatialPatcher`, `MatchedTemporalPatcher`,
`MatchedSpatioTemporalPatcher`, and `MatchedField`: each returns a
plain, JSON-serializable dict that carries the inner single-source
patcher's config envelope plus the matched layer's own fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from geopatcher import spatial, temporal
from geopatcher._src.matched import (
    MatchedField,
    MatchedSpatialPatcher,
    MatchedSpatioTemporalPatcher,
    MatchedTemporalPatcher,
)
from geopatcher._src.spatial.patcher import SpatialPatcher
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.temporal.patcher import TemporalPatcher


@dataclass
class _StubField:
    """Minimal `Field` stand-in — enough for `MatchedField` construction."""

    name: str

    @property
    def domain(self) -> Any:
        return None

    def select(self, indexer: Any) -> str:
        return f"{self.name}@{indexer}"

    def with_data(self, array: Any) -> Any:
        return array


def _spatial_patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.Sum(),
    )


def _temporal_patcher() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=4),
        sampler=temporal.sampler.RegularStride(step=4),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )


def _assert_envelope(entry: dict[str, Any], cls_name: str) -> None:
    assert entry["class"] == cls_name
    assert isinstance(entry["config"], dict)


class TestMatchedPatcherGetConfig:
    def test_spatial(self) -> None:
        patcher = MatchedSpatialPatcher(
            primary=_spatial_patcher(),
            secondary_aggregators={"s2": spatial.aggregation.Mean()},
        )
        cfg = patcher.get_config()
        assert isinstance(cfg, dict)
        _assert_envelope(cfg["primary"], "SpatialPatcher")
        # The inner patcher's own envelope shape flows through.
        _assert_envelope(
            cfg["primary"]["config"]["geometry"], "spatial.geometry.Rectangular"
        )
        _assert_envelope(cfg["secondary_aggregators"]["s2"], "spatial.aggregation.Mean")
        json.dumps(cfg)  # must be JSON-serializable

    def test_temporal(self) -> None:
        patcher = MatchedTemporalPatcher(
            primary=_temporal_patcher(),
            secondary_aggregators={"s2": temporal.aggregation.Mean()},
        )
        cfg = patcher.get_config()
        assert isinstance(cfg, dict)
        _assert_envelope(cfg["primary"], "TemporalPatcher")
        _assert_envelope(
            cfg["primary"]["config"]["aggregation"], "temporal.aggregation.Mean"
        )
        _assert_envelope(
            cfg["secondary_aggregators"]["s2"], "temporal.aggregation.Mean"
        )
        json.dumps(cfg)

    def test_spatiotemporal(self) -> None:
        patcher = MatchedSpatioTemporalPatcher(
            primary=SpatioTemporalPatcher(
                spatial=_spatial_patcher(),
                temporal=_temporal_patcher(),
                coupling="product",
            ),
            secondary_aggregators={"s2": temporal.aggregation.Mean()},
        )
        cfg = patcher.get_config()
        assert isinstance(cfg, dict)
        _assert_envelope(cfg["primary"], "SpatioTemporalPatcher")
        assert cfg["primary"]["config"]["coupling"] == "product"
        _assert_envelope(
            cfg["secondary_aggregators"]["s2"], "temporal.aggregation.Mean"
        )
        json.dumps(cfg)

    def test_empty_secondaries(self) -> None:
        cfg = MatchedSpatialPatcher(primary=_spatial_patcher()).get_config()
        assert cfg["secondary_aggregators"] == {}
        json.dumps(cfg)


class TestMatchedFieldGetConfig:
    def test_names_and_classes(self) -> None:
        mfield = MatchedField(
            primary=_StubField("geo"),
            secondaries={"s2": _StubField("s2"), "leo": _StubField("leo")},
            coreg={"s2": lambda raw, prim: raw, "leo": lambda raw, prim: raw},
            valid_mask=False,
        )
        cfg = mfield.get_config()
        assert isinstance(cfg, dict)
        assert cfg["primary"] == {"class": "_StubField"}
        assert set(cfg["secondaries"]) == {"s2", "leo"}
        assert cfg["secondaries"]["s2"] == {"class": "_StubField"}
        assert cfg["valid_mask"] is False
        json.dumps(cfg)

    def test_forbid_in_yaml(self) -> None:
        # Fields / coreg callables are not reconstructable from config.
        assert MatchedField.forbid_in_yaml is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
