"""Tests for `geotoolz.indices`.

Three layers:

1. **Tier-A math** — analytic ground truth on toy ndarrays. The
   primitives in ``geotoolz.indices._src.array`` are pure functions of
   numpy arrays; they should agree with hand-computed formulas to
   floating-point precision.
2. **Tier-B carrier round-trip** — ``transform``, ``crs`` and
   ``fill_value_default`` survive ``op(gt)``.
3. **Hydra-zen ``builds()`` round-trip** — every Operator's
   ``get_config()`` re-instantiates an equivalent operator.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.indices import (
    ARVI,
    BAIS2,
    BSI,
    CIRI,
    EVI,
    EVI2,
    GCI,
    MNDWI,
    NBR,
    NBR2,
    NDBI,
    NDMI,
    NDSI,
    NDVI,
    NDWI,
    SAVI,
    AppendIndex,
    ClayMinerals,
    IronOxide,
    NormalizedDifference,
    arvi,
    bais2,
    bsi,
    ciri,
    clay_minerals,
    dNBR,
    evi,
    evi2,
    gci,
    iron_oxide,
    kNDVI,
    kndvi,
    mndwi,
    nbr,
    nbr2,
    ndbi,
    ndmi,
    ndsi,
    ndvi,
    ndwi_mcfeeters,
    normalized_difference,
    savi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    """Build a small `GeoTensor` with stable transform/CRS for round-trip tests."""
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )


@pytest.fixture
def reflectance_4band() -> GeoTensor:
    """``(B, G, R, NIR)`` reflectance carrier, 4 x 8 x 8."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.05, 0.6, size=(4, 8, 8)).astype(np.float32)
    return _toy_geotensor(arr)


@pytest.fixture
def reflectance_7band() -> GeoTensor:
    """``(B, G, R, NIR, SWIR1, SWIR2, ??)`` — 7-band stub for NDBI / NBR."""
    rng = np.random.default_rng(1)
    arr = rng.uniform(0.05, 0.6, size=(7, 8, 8)).astype(np.float32)
    return _toy_geotensor(arr)


# ---------------------------------------------------------------------------
# Tier-A — primitive math
# ---------------------------------------------------------------------------


def test_normalized_difference_matches_formula() -> None:
    arr = np.array(
        [
            [[0.8, 0.4]],
            [[0.2, 0.6]],
        ],
        dtype=np.float32,
    )  # (2, 1, 2): band0=high, band1=low
    out = normalized_difference(arr, 0, 1, axis=0, eps=0.0)
    expected = np.array([[0.6, -0.2]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_normalized_difference_eps_shadows_zero_division() -> None:
    arr = np.zeros((2, 1, 1), dtype=np.float32)
    # With eps=0 we'd hit 0/0 -> nan; the default eps shadows it to 0.
    out = normalized_difference(arr, 0, 1, eps=1e-10)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 0.0, atol=1e-9)


def test_normalized_difference_zero_eps_gives_inf_or_nan() -> None:
    arr = np.zeros((2, 1, 1), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = normalized_difference(arr, 0, 1, eps=0.0)
    assert not np.isfinite(out).all()  # the user opted in to the singularity


def test_ndvi_savi_evi_match_hand_computed() -> None:
    # Tiny toy array where the math is trivial.
    # bands ordered (B, G, R, NIR).
    arr = np.array(
        [
            [[0.05]],  # B = 0.05
            [[0.10]],  # G
            [[0.20]],  # R
            [[0.60]],  # NIR
        ],
        dtype=np.float32,
    )
    # NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
    np.testing.assert_allclose(ndvi(arr, 3, 2, eps=0.0), [[0.5]], rtol=1e-6)

    # SAVI (L=0.5) = (0.6 - 0.2)/(0.6 + 0.2 + 0.5) * 1.5 = 0.4/1.3 * 1.5
    np.testing.assert_allclose(savi(arr, 3, 2, L=0.5), [[0.4 / 1.3 * 1.5]], rtol=1e-6)

    # EVI = 2.5 * (0.6 - 0.2) / (0.6 + 6*0.2 - 7.5*0.05 + 1)
    #     = 2.5 * 0.4 / (0.6 + 1.2 - 0.375 + 1)
    #     = 1.0 / 2.425
    np.testing.assert_allclose(evi(arr, 3, 2, 0), [[1.0 / 2.425]], rtol=1e-6)


def test_ndwi_ndbi_nbr_match_normalized_difference() -> None:
    """Named indices should match the generic primitive with the right bands."""
    arr = np.array(
        [
            [[0.10]],  # 0: B
            [[0.30]],  # 1: G
            [[0.20]],  # 2: R
            [[0.55]],  # 3: NIR
            [[0.40]],  # 4
            [[0.45]],  # 5: SWIR1
            [[0.35]],  # 6: SWIR2
        ],
        dtype=np.float32,
    )
    # NDWI (Green - NIR) / (Green + NIR)
    np.testing.assert_allclose(
        ndwi_mcfeeters(arr, 1, 3, eps=0.0),
        normalized_difference(arr, 1, 3, eps=0.0),
    )
    # NDBI (SWIR - NIR) / (SWIR + NIR)
    np.testing.assert_allclose(
        ndbi(arr, 5, 3, eps=0.0),
        normalized_difference(arr, 5, 3, eps=0.0),
    )
    # NBR (NIR - SWIR2) / (NIR + SWIR2)
    np.testing.assert_allclose(
        nbr(arr, 3, 6, eps=0.0),
        normalized_difference(arr, 3, 6, eps=0.0),
    )


def test_additional_indices_match_hand_computed() -> None:
    arr = np.array(
        [
            [[0.05]],  # B
            [[0.10]],  # G
            [[0.20]],  # R
            [[0.60]],  # NIR
            [[0.30]],  # RE1
            [[0.40]],  # RE2 / SWIR1
            [[0.35]],  # SWIR2
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        evi2(arr, 3, 2, eps=0.0), [[2.5 * 0.4 / 2.08]], rtol=1e-6
    )
    np.testing.assert_allclose(arvi(arr, 3, 2, 0, eps=0.0), [[0.25 / 0.95]])
    np.testing.assert_allclose(gci(arr, 3, 1, eps=0.0), [[5.0]])
    np.testing.assert_allclose(
        kndvi(arr, 3, 2, eps=0.0), np.tanh([[0.5**2]]), rtol=1e-6
    )
    np.testing.assert_allclose(
        mndwi(arr, 1, 5, eps=0.0),
        normalized_difference(arr, 1, 5, eps=0.0),
    )
    np.testing.assert_allclose(
        ndmi(arr, 3, 5, eps=0.0),
        normalized_difference(arr, 3, 5, eps=0.0),
    )
    np.testing.assert_allclose(
        ndsi(arr, 1, 5, eps=0.0),
        normalized_difference(arr, 1, 5, eps=0.0),
    )
    np.testing.assert_allclose(
        nbr2(arr, 5, 6, eps=0.0),
        normalized_difference(arr, 5, 6, eps=0.0),
    )
    np.testing.assert_allclose(
        bsi(arr, 0, 2, 3, 5, eps=0.0),
        [[((0.4 + 0.2) - (0.6 + 0.05)) / ((0.4 + 0.2) + (0.6 + 0.05))]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(iron_oxide(arr, 2, 0, eps=0.0), [[4.0]])
    np.testing.assert_allclose(clay_minerals(arr, 5, 6, eps=0.0), [[0.4 / 0.35]])
    np.testing.assert_allclose(ciri(arr, 6), [[0.35]])
    assert np.isfinite(bais2(arr, 2, 4, 5, 3, 6)).all()


def test_savi_l0_equals_ndvi() -> None:
    rng = np.random.default_rng(7)
    arr = rng.uniform(0.05, 0.6, size=(4, 5, 5)).astype(np.float32)
    np.testing.assert_allclose(
        savi(arr, 3, 2, L=0.0),
        ndvi(arr, 3, 2, eps=0.0),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Tier-B — Operator + GeoTensor round-trip
# ---------------------------------------------------------------------------


def test_ndvi_preserves_transform_and_crs(reflectance_4band: GeoTensor) -> None:
    out = NDVI(nir_idx=3, red_idx=2)(reflectance_4band)
    assert isinstance(out, GeoTensor)
    assert out.transform == reflectance_4band.transform
    assert str(out.crs) == "EPSG:32629"
    # Band axis collapsed -> 2D.
    assert out.shape == (8, 8)
    # Values in [-1, +1] for non-negative reflectance.
    assert np.all(np.asarray(out) >= -1.0)
    assert np.all(np.asarray(out) <= 1.0)


def test_normalized_difference_op_matches_named_subclass(
    reflectance_4band: GeoTensor,
) -> None:
    via_named = NDVI(nir_idx=3, red_idx=2, eps=0.0)(reflectance_4band)
    via_generic = NormalizedDifference(a_idx=3, b_idx=2, eps=0.0)(reflectance_4band)
    np.testing.assert_allclose(np.asarray(via_named), np.asarray(via_generic))


def test_all_indices_run_without_crashing(reflectance_7band: GeoTensor) -> None:
    """Smoke test every Operator's _apply path."""
    for op in [
        ARVI(),
        BAIS2(red_idx=2, red_edge1_idx=4, red_edge2_idx=5, nir_idx=3, swir2_idx=6),
        BSI(),
        ClayMinerals(),
        EVI2(),
        NDVI(),
        NDWI(),
        GCI(),
        IronOxide(),
        kNDVI(),
        MNDWI(),
        NDBI(),
        NDMI(),
        NDSI(),
        NBR(),
        NBR2(),
        SAVI(),
        EVI(),
        CIRI(cirrus_idx=6),
        NormalizedDifference(a_idx=3, b_idx=2),
    ]:
        out = op(reflectance_7band)
        assert isinstance(out, GeoTensor)
        assert out.shape == (8, 8)
        assert out.transform == reflectance_7band.transform


def test_band_name_resolution_uses_geotensor_descriptions() -> None:
    rng = np.random.default_rng(2)
    arr = rng.uniform(0.05, 0.6, size=(4, 5, 5)).astype(np.float32)
    gt = _toy_geotensor(arr)
    gt.attrs["descriptions"] = ("B02", "B03", "B04", "B08")

    via_names = NDVI(red="B04", nir="B08", eps=0.0)(gt)
    via_indices = NDVI(red_idx=2, nir_idx=3, eps=0.0)(gt)

    np.testing.assert_allclose(np.asarray(via_names), np.asarray(via_indices))


def test_band_name_resolution_missing_name_raises() -> None:
    gt = _toy_geotensor(np.ones((2, 2, 2), dtype=np.float32))
    gt.attrs["descriptions"] = ("B02", "B03")

    with pytest.raises(ValueError, match="Band 'B04' was not found"):
        NDVI(red="B04", nir="B03")(gt)


def test_dnbr_subtracts_matching_geotensors() -> None:
    pre = _toy_geotensor(np.full((3, 3), 0.7, dtype=np.float32))
    post = _toy_geotensor(np.full((3, 3), 0.2, dtype=np.float32))

    out = dNBR()(pre, post)

    np.testing.assert_allclose(np.asarray(out), 0.5)
    assert out.transform == pre.transform
    assert out.crs == pre.crs


def test_dnbr_raises_on_grid_mismatch() -> None:
    pre = _toy_geotensor(np.ones((3, 3), dtype=np.float32))
    post = _toy_geotensor(np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="share shape, transform, and CRS"):
        dNBR()(pre, post)


def test_append_index_concatenates_back(reflectance_4band: GeoTensor) -> None:
    op = AppendIndex(index_op=NDVI(nir_idx=3, red_idx=2))
    out = op(reflectance_4band)
    assert isinstance(out, GeoTensor)
    assert out.shape == (5, 8, 8)  # original 4 bands + 1 NDVI channel
    # The new last channel should equal a direct NDVI call.
    expected_ndvi = np.asarray(NDVI(nir_idx=3, red_idx=2)(reflectance_4band))
    np.testing.assert_allclose(np.asarray(out)[-1], expected_ndvi, rtol=1e-6)


# ---------------------------------------------------------------------------
# Hydra-zen round-trip
# ---------------------------------------------------------------------------


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize(
    "op",
    [
        NDVI(nir_idx=7, red_idx=3, eps=1e-8),
        NDWI(green_idx=2, nir_idx=7),
        NDBI(swir_idx=10, nir_idx=7),
        NBR(nir_idx=7, swir2_idx=11),
        SAVI(nir_idx=7, red_idx=3, L=0.3),
        EVI(nir_idx=7, red_idx=3, blue_idx=1, G=2.5, C1=6.0, C2=7.5, L=1.0),
        NormalizedDifference(a_idx=4, b_idx=2, axis=0, eps=1e-9),
    ],
)
def test_indices_hydra_zen_roundtrip(op: object) -> None:
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_append_index_get_config_is_jsonable() -> None:
    """`AppendIndex.get_config()` should emit nested {class, config} for
    its inner Operator, not the raw instance."""
    import json

    op = AppendIndex(index_op=NDVI(nir_idx=7, red_idx=3), axis=0)
    cfg = op.get_config()
    # Round-trips through JSON without choking on raw Operator instances.
    encoded = json.dumps(cfg)
    decoded = json.loads(encoded)
    assert decoded == {
        "index_op": {
            "class": "NDVI",
            "config": {"nir_idx": 7, "red_idx": 3, "axis": 0, "eps": 1e-10},
        },
        "axis": 0,
    }
    # Manual reconstruction from the nested config works.
    inner_cls_name = decoded["index_op"]["class"]
    assert inner_cls_name == "NDVI"
    restored = AppendIndex(
        index_op=NDVI(**decoded["index_op"]["config"]), axis=decoded["axis"]
    )
    assert restored.get_config() == cfg
