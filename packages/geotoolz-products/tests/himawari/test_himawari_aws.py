"""``geoproducts.himawari.aws`` against a fake S3 endpoint (no network)."""

from __future__ import annotations

import bz2
from datetime import UTC, datetime

import pytest
from _s3_fake import listing

from geoproducts.himawari import aws


SEG = "HS_H09_20261007_0300_B13_FLDK_R20_S0510.DAT.bz2"
CMSK = "AHI-CMSK_v1r1_h09_s202610070300214_e202610070309408_c202610070314065.nc"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "_BACKOFF_S", 0.0)


def _fldk(slot: str, band: int, segment: int) -> str:
    day, hhmm = slot.split("_")
    return (
        f"AHI-L1b-FLDK/{day[:4]}/{day[4:6]}/{day[6:]}/{hhmm}/"
        f"HS_H09_{slot}_B{band:02d}_FLDK_R20_S{segment:02d}10.DAT.bz2"
    )


class TestNames:
    def test_parse_segment(self) -> None:
        item = aws.parse_key(f"AHI-L1b-FLDK/2026/10/07/0300/{SEG}", size=7)
        assert isinstance(item, aws.Segment)
        assert item.satellite == "H09"
        assert item.slot == datetime(2026, 10, 7, 3, 0, tzinfo=UTC)
        assert (item.band, item.area, item.resolution_km) == ("B13", "FLDK", 2.0)
        assert (item.segment, item.total_segments, item.size) == (5, 10, 7)
        assert item.name == SEG
        assert item.url == (
            f"https://noaa-himawari9.s3.amazonaws.com/AHI-L1b-FLDK/2026/10/07/0300/{SEG}"
        )

    @pytest.mark.parametrize(
        ("name", "area", "resolution"),
        [
            ("HS_H09_20261007_0300_B03_JP02_R05_S0101.DAT.bz2", "JP02", 0.5),
            ("HS_H08_20200101_0000_B01_R301_R10_S0101.DAT", "R301", 1.0),
        ],
    )
    def test_parse_sub_areas(self, name, area, resolution) -> None:
        item = aws.parse_key(name)
        assert (item.area, item.resolution_km) == (area, resolution)

    def test_parse_l2(self) -> None:
        item = aws.parse_key(CMSK)
        assert isinstance(item, aws.L2File)
        assert (item.product, item.satellite) == ("CMSK", "H09")
        assert item.start == datetime(2026, 10, 7, 3, 0, 21, 400_000, tzinfo=UTC)
        assert item.end > item.start and item.created > item.end

    def test_other_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a Himawari"):
            aws.parse_key("OR_ABI-L1b-RadC-M6C13_G19_s1_e1_c1.nc")

    @pytest.mark.parametrize("satellite", ["H09", "h9", 9, "Himawari-9", "HIMAWARI9"])
    def test_bucket_accepts_satellite_spellings(self, satellite) -> None:
        assert aws.bucket(satellite) == "noaa-himawari9"

    @pytest.mark.parametrize("satellite", ["H07", 10, "G19", ""])
    def test_bucket_rejects_other_satellites(self, satellite) -> None:
        with pytest.raises(ValueError, match="H08 or H09"):
            aws.bucket(satellite)


class TestListing:
    def test_filters_by_band_and_segment_across_slots(self, fake_s3) -> None:
        s0300 = [_fldk("20261007_0300", b, s) for b in (3, 13) for s in (4, 5, 6)]
        s0310 = [_fldk("20261007_0310", 13, 5)]
        fake = fake_s3(
            pages={
                "AHI-L1b-FLDK/2026/10/07/0300/": [listing(s0300)],
                "AHI-L1b-FLDK/2026/10/07/0310/": [listing(s0310)],
            }
        )
        out = aws.list_segments(
            start=datetime(2026, 10, 7, 3, 7),
            end=datetime(2026, 10, 7, 3, 20),
            band=13,
            segments=[5, 6],
        )
        assert [(f.slot.minute, f.segment) for f in out] == [(0, 5), (0, 6), (10, 5)]
        assert all(f.band == "B13" for f in out)
        assert len(fake.urls) == 2  # one listing per slot
        assert "noaa-himawari9" in fake.urls[0]

    def test_japan_sub_scans_and_area_filter(self, fake_s3) -> None:
        keys = [
            f"AHI-L1b-Japan/2026/10/07/0300/HS_H09_20261007_0300_B13_JP0{n}_R20_S0101.DAT.bz2"
            for n in (1, 2, 3, 4)
        ]
        fake_s3(pages={"AHI-L1b-Japan/2026/10/07/0300/": [listing(keys)]})
        start = datetime(2026, 10, 7, 3, 0)
        assert [f.area for f in aws.list_segments(start=start, sector="Japan")] == [
            "JP01",
            "JP02",
            "JP03",
            "JP04",
        ]
        only = aws.list_segments(start=start, sector="Japan", area="JP03", segments=1)
        assert [f.area for f in only] == ["JP03"]

    def test_default_window_is_the_start_slot(self, fake_s3) -> None:
        fake = fake_s3()
        aws.list_segments(start=datetime(2026, 10, 7, 3, 7))
        assert len(fake.urls) == 1
        assert "0300%2F" in fake.urls[0]

    def test_listing_skips_foreign_keys(self, fake_s3) -> None:
        prefix = "AHI-L1b-FLDK/2026/10/07/0300/"
        fake_s3(pages={prefix: [listing([f"{prefix}README.txt", f"{prefix}{SEG}"])]})
        assert len(aws.list_segments(start=datetime(2026, 10, 7, 3))) == 1

    def test_himawari_8_archive(self, fake_s3) -> None:
        fake = fake_s3()
        aws.list_segments(start=datetime(2020, 1, 1), satellite="H08")
        assert fake.urls[0].startswith("https://noaa-himawari8.")
        assert "AHI-L1b-FLDK%2F2020%2F01%2F01%2F0000%2F" in fake.urls[0]

    def test_l2_listing(self, fake_s3) -> None:
        prefix = "AHI-L2-FLDK-Clouds/2026/10/07/0300/"
        chgt = CMSK.replace("CMSK", "CHGT")
        fake_s3(pages={prefix: [listing([prefix + CMSK, prefix + chgt, prefix + SEG])]})
        out = aws.list_l2(start=datetime(2026, 10, 7, 3, 0))
        assert [f.product for f in out] == ["CMSK"]
        heights = aws.list_l2(start=datetime(2026, 10, 7, 3, 0), product="CHGT")
        assert [f.product for f in heights] == ["CHGT"]
        assert aws.list_l2(start=datetime(2026, 10, 7, 3, 1)) == []

    def test_l2_default_window_spans_into_the_next_slot(self, fake_s3) -> None:
        later = CMSK.replace("s202610070300214", "s202610070310214")
        fake = fake_s3(
            pages={
                "AHI-L2-FLDK-Clouds/2026/10/07/0310/": [
                    listing([f"AHI-L2-FLDK-Clouds/2026/10/07/0310/{later}"])
                ]
            }
        )
        out = aws.list_l2(start=datetime(2026, 10, 7, 3, 7))
        assert [f.start.minute for f in out] == [10]
        assert len(fake.urls) == 2  # the 03:00 and 03:10 slots

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"end": datetime(2026, 10, 7, 3)}, "must be after"),
            ({"sector": "Mesoscale"}, "sector must be one of"),
            ({"band": 17}, "band must be 1-16"),
        ],
    )
    def test_bad_arguments(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            aws.list_segments(start=datetime(2026, 10, 7, 3), **kwargs)


class TestDownload:
    def test_download_keeps_the_compressed_file(self, fake_s3, tmp_path) -> None:
        key = f"AHI-L1b-FLDK/2026/10/07/0300/{SEG}"
        fake_s3(bodies={key: bz2.compress(b"hsd")})
        path = aws.download(key, tmp_path)
        assert path.name == SEG
        assert bz2.decompress(path.read_bytes()) == b"hsd"

    def test_download_decompressed(self, fake_s3, tmp_path) -> None:
        key = f"AHI-L1b-FLDK/2026/10/07/0300/{SEG}"
        fake = fake_s3(bodies={key: bz2.compress(b"hsd" * 1000)})
        path = aws.download(aws.parse_key(key), tmp_path, decompress=True)
        assert path.name == SEG.removesuffix(".bz2")
        assert path.read_bytes() == b"hsd" * 1000
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
        # Reused unless overwrite.
        assert aws.download(key, tmp_path, decompress=True) == path
        assert len(fake.urls) == 1

    def test_decompress_reuses_and_keeps_a_local_archive(
        self, fake_s3, tmp_path
    ) -> None:
        key = f"AHI-L1b-FLDK/2026/10/07/0300/{SEG}"
        fake = fake_s3(bodies={key: bz2.compress(b"hsd")})
        packed = aws.download(key, tmp_path)
        plain = aws.download(key, tmp_path, decompress=True)
        assert len(fake.urls) == 1  # decompressed from the local archive
        assert packed.exists() and plain.read_bytes() == b"hsd"
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            [packed.name, plain.name]
        )

    def test_corrupt_archive_leaves_nothing(self, fake_s3, tmp_path) -> None:
        key = f"AHI-L1b-FLDK/2026/10/07/0300/{SEG}"
        fake_s3(bodies={key: b"not bzip2"})
        with pytest.raises(OSError):
            aws.download(key, tmp_path, decompress=True)
        assert list(tmp_path.iterdir()) == []
