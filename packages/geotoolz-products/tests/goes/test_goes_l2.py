"""``geoproducts.goes.L2Reader`` on synthetic ABI L2 product files."""

from __future__ import annotations

import numpy as np
import pytest
from _goes_abi import l2_name
from rasterio.windows import Window

from geoproducts import goes
from geoproducts.goes import constants
from geoproducts.goes.l2 import product_code


SHAPE = (4, 6)


def _u8(values, **attrs):
    return np.asarray(values, dtype=np.uint8), {"_FillValue": np.uint8(255), **attrs}


def _packed(values, *, scale, offset, units, fill=65535):
    attrs = {
        "_FillValue": np.uint16(fill),
        "scale_factor": np.float32(scale),
        "add_offset": np.float32(offset),
        "units": np.bytes_(units.encode()),
    }
    return np.asarray(values, dtype=np.uint16), attrs


def _acm(l2_file):
    bcm = np.zeros(SHAPE, dtype=np.uint8)
    bcm[0, :3] = 1
    bcm[3, 5] = 255
    acm = bcm * 3
    acm[1, 0] = 1  # probably clear
    acm[3, 5] = 255
    meanings = b"clear probably_clear probably_cloudy cloudy"
    return l2_file(
        {
            "BCM": _u8(bcm),
            "ACM": _u8(
                acm,
                flag_values=np.arange(4, dtype=np.uint8),
                flag_meanings=np.bytes_(meanings),
            ),
            "Cloud_Probabilities": _packed(
                np.full(SHAPE, 30000), scale=1.5e-5, offset=0.0, units="1"
            ),
            "DQF": _u8(np.zeros(SHAPE)),
        },
        code="ACM",
    )


def _mcmip(l2_file, channels=("C01", "C02", "C03", "C13")):
    variables = {}
    extra = {}
    for i, ch in enumerate(channels):
        emissive = ch in constants.EMISSIVE_CHANNELS
        stored = np.full(SHAPE, 1000 * (i + 1), dtype=np.uint16).view(np.int16)
        variables[f"CMI_{ch}"] = (
            stored,
            {
                "_FillValue": np.int16(-1),
                "_Unsigned": np.bytes_(b"true"),
                "scale_factor": np.float32(0.06 if emissive else 3.1746e-4),
                "add_offset": np.float32(89.62 if emissive else 0.0),
                "units": np.bytes_(b"K" if emissive else b"1"),
            },
        )
        variables[f"DQF_{ch}"] = (np.zeros(SHAPE, dtype=np.int8), {})
        extra[f"band_wavelength_{ch}"] = np.array([0.47 + i], dtype=np.float32)
    return l2_file(variables, code="MCMIP", extra=extra)


class TestProducts:
    @pytest.mark.parametrize(
        ("code", "sector", "expected"),
        [
            ("ACM", "C", "ACM"),
            ("ACM", "M1", "ACM"),
            ("ACHA2KM", "F", "ACHA2KM"),
            ("MCMIP", "M2", "MCMIP"),
        ],
    )
    def test_product_code_from_the_file_name(self, code, sector, expected) -> None:
        assert product_code(l2_name(code, sector)) == expected

    def test_non_l2_name_has_no_code(self) -> None:
        assert product_code("OR_ABI-L1b-RadC-M6C13_G19_x.nc") == ""


class TestClearSkyMask:
    def test_default_bands_stay_integer(self, l2_file) -> None:
        reader = goes.L2Reader(_acm(l2_file))
        assert reader.product == "ACM"
        assert reader.bands == ("BCM", "ACM")
        assert reader.dtype == np.uint8
        assert reader.fill_value_default == 255
        out = reader.load()
        assert out.shape == (2, *SHAPE)
        assert np.asarray(out)[0, 0, :3].tolist() == [1, 1, 1]
        assert np.asarray(out)[1, 1, 0] == constants.ACM_PROBABLY_CLEAR
        assert out.attrs["product"] == "ACM"

    def test_flag_meanings(self, l2_file) -> None:
        reader = goes.L2Reader(_acm(l2_file))
        assert reader.flags("ACM") == {
            0: "clear",
            1: "probably_clear",
            2: "probably_cloudy",
            3: "cloudy",
        }
        assert reader.flags() == {}  # BCM declares none here

    def test_mixing_a_packed_band_decodes_to_float(self, l2_file) -> None:
        reader = goes.L2Reader(_acm(l2_file), variables=["BCM", "Cloud_Probabilities"])
        out = np.asarray(reader.load())
        assert out.dtype == np.float32
        assert np.isnan(out[0, 3, 5])  # BCM fill → NaN
        assert out[1, 0, 0] == pytest.approx(30000 * 1.5e-5)

    def test_available_variables_and_quality(self, l2_file) -> None:
        reader = goes.L2Reader(_acm(l2_file))
        assert set(reader.available_variables) == {
            "BCM",
            "ACM",
            "Cloud_Probabilities",
            "DQF",
        }
        assert reader.quality.bands == ("DQF",)
        assert reader.quality.transform == reader.transform


class TestPackedProducts:
    def test_single_variable_product_defaults_to_it(self, l2_file) -> None:
        values = np.arange(24).reshape(SHAPE) * 100
        values[0, 0] = 65535
        path = l2_file(
            {
                "HT": _packed(values, scale=0.3052, offset=0.0, units="m"),
                "DQF": _u8(np.zeros(SHAPE)),
            },
            code="ACHA",
        )
        reader = goes.L2Reader(path)
        assert reader.bands == ("HT",)
        out = reader.load()
        assert out.dtype == np.float32
        assert out.attrs["units"] == ("m",)
        assert np.isnan(np.asarray(out)[0, 0, 0])
        np.testing.assert_allclose(
            np.asarray(out)[0, 1:, :], values[1:, :] * np.float32(0.3052), rtol=1e-6
        )

    def test_sounding_products_map_to_dqf_overall(self, l2_file) -> None:
        path = l2_file(
            {
                "CAPE": _packed(
                    np.ones(SHAPE), scale=0.076, offset=0.0, units="J kg-1"
                ),
                "LI": _packed(np.ones(SHAPE), scale=0.00076, offset=-10.0, units="K"),
                "DQF_Overall": _u8(np.zeros(SHAPE)),
            },
            code="DSI",
        )
        reader = goes.L2Reader(path)
        assert reader.bands == ("CAPE", "LI")
        assert reader.quality.bands == ("DQF_Overall",)

    def test_lst_default_skips_the_quality_bit_field(self, l2_file) -> None:
        path = l2_file(
            {
                "LST": _packed(np.ones(SHAPE), scale=0.0025, offset=190.0, units="K"),
                "PQI": (np.zeros(SHAPE, dtype=np.uint16), {}),
            },
            code="LST",
        )
        assert goes.L2Reader(path).bands == ("LST",)

    def test_float_variable_with_a_fill(self, l2_file) -> None:
        power = np.full(SHAPE, -9.0, dtype=np.float32)
        power[2, 2] = 125.5
        path = l2_file(
            {
                "Mask": (
                    np.full(SHAPE, 30, dtype=np.int16),
                    {"_FillValue": np.int16(-99)},
                ),
                "Power": (power, {"_FillValue": np.float32(-9.0)}),
            },
            code="FDC",
        )
        assert goes.L2Reader(path).bands == ("Mask",)
        out = np.asarray(goes.L2Reader(path, variables="Power").load())[0]
        assert out[2, 2] == pytest.approx(125.5)
        assert np.isnan(out).sum() == out.size - 1


class TestImagery:
    def test_mcmip_bands_are_named_by_channel(self, l2_file) -> None:
        reader = goes.L2Reader(_mcmip(l2_file))
        assert reader.product == "MCMIP"
        assert reader.variables == ("CMI_C01", "CMI_C02", "CMI_C03", "CMI_C13")
        assert reader.bands == ("C01", "C02", "C03", "C13")
        out = reader.load()
        assert out.attrs["wavelengths"] == pytest.approx(
            (470.0, 1470.0, 2470.0, 3470.0)
        )
        # _Unsigned storage: 4000 counts read as unsigned, then scaled.
        assert np.asarray(out)[3, 0, 0] == pytest.approx(4000 * 0.06 + 89.62, rel=1e-5)
        assert reader.quality.bands == ("DQF_C01", "DQF_C02", "DQF_C03", "DQF_C13")

    def test_cmip_band_is_named_from_band_id(self, l2_file) -> None:
        path = l2_file(
            {
                "CMI": (
                    np.full(SHAPE, 100, dtype=np.int16),
                    {"_FillValue": np.int16(-1), "scale_factor": np.float32(0.01)},
                ),
                "DQF": (np.zeros(SHAPE, dtype=np.int8), {}),
            },
            code="CMIP",
            extra={
                "band_id": np.array([13], dtype=np.int8),
                "band_wavelength": np.array([10.3], dtype=np.float32),
            },
        )
        reader = goes.L2Reader(path)
        assert reader.bands == ("C13",)
        assert reader.load().attrs["wavelengths"] == (pytest.approx(10300.0),)


class TestErrors:
    def test_unknown_variable_lists_the_choices(self, l2_file) -> None:
        with pytest.raises(ValueError, match=r"no variable 'HT'.*BCM"):
            goes.L2Reader(_acm(l2_file), variables="HT")

    def test_empty_variable_list(self, l2_file) -> None:
        with pytest.raises(ValueError, match="at least one"):
            goes.L2Reader(_acm(l2_file), variables=[])

    def test_variable_off_the_grid(self, l2_file) -> None:
        path = l2_file(
            {"HT": _packed(np.ones(SHAPE), scale=1.0, offset=0.0, units="m")},
            extra={"profile": np.ones((3, 3), dtype=np.float32)},
        )
        with pytest.raises(ValueError, match="not the fixed grid"):
            goes.L2Reader(path, variables="profile")

    def test_l1b_reader_points_l2_files_to_l2reader(self, l2_file) -> None:
        with pytest.raises(ValueError, match=r"goes\.L2Reader"):
            goes.Reader(_acm(l2_file))

    def test_bit_field_flags_are_refused(self, l2_file) -> None:
        path = l2_file(
            {
                "PQI": (
                    np.zeros(SHAPE, dtype=np.uint16),
                    {
                        "flag_values": np.array([0, 1, 0, 4], dtype=np.uint16),
                        "flag_meanings": np.bytes_(b"a b c"),
                    },
                )
            }
        )
        with pytest.raises(ValueError, match="flag_masks"):
            goes.L2Reader(path).flags()

    def test_no_data_variables(self, l2_file) -> None:
        path = l2_file({"DQF": _u8(np.zeros(SHAPE))})
        with pytest.raises(ValueError, match="no 2-D data variables"):
            goes.L2Reader(path)

    def test_no_quality_flags(self, l2_file) -> None:
        path = l2_file(
            {"HT": _packed(np.ones(SHAPE), scale=1.0, offset=0.0, units="m")}
        )
        with pytest.raises(ValueError, match="no quality flags"):
            _ = goes.L2Reader(path).quality


def test_windowed_read_and_repr(l2_file) -> None:
    reader = goes.L2Reader(_acm(l2_file))
    chip = np.asarray(reader.read_from_window(Window(4, 2, 4, 3)))
    assert chip.shape == (2, 3, 4)
    assert (chip[:, :, 2:] == 255).all()  # past the right edge: fill
    assert "product='ACM'" in repr(reader)
