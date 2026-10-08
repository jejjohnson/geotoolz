"""``geoproducts.goes.aws`` against in-memory buckets (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from geoproducts.goes import aws


RADC = "OR_ABI-L1b-RadC-M6C13_G19_s20262801201178_e20262801203551_c20262801203578.nc"
RADM = "OR_ABI-L1b-RadM2-M6C01_G18_s20262801200556_e20262801201025_c20262801201053.nc"


def _key(hour: int, minute: int, channel: int, *, product: str = "RadC") -> str:
    start = f"2026280{hour:02d}{minute:02d}178"
    return (
        f"ABI-L1b-{product[:4]}/2026/280/{hour:02d}/OR_ABI-L1b-{product}-M6"
        f"C{channel:02d}_G19_s{start}_e{start}_c{start}.nc"
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "_BACKOFF_S", 0.0)


class TestNames:
    def test_parse_conus_key(self) -> None:
        item = aws.parse_key(f"ABI-L1b-RadC/2026/280/12/{RADC}", size=5)
        assert item.satellite == "G19"
        assert item.product == "ABI-L1b-RadC"
        assert item.sector is None
        assert item.mode == 6
        assert item.channel == "C13"
        assert item.start == datetime(2026, 10, 7, 12, 1, 17, 800_000, UTC)
        assert item.end == datetime(2026, 10, 7, 12, 3, 55, 100_000, UTC)
        assert item.size == 5
        assert item.name == RADC
        assert item.url == (
            f"https://noaa-goes19.s3.amazonaws.com/ABI-L1b-RadC/2026/280/12/{RADC}"
        )

    def test_parse_mesoscale_key(self) -> None:
        item = aws.parse_key(RADM)
        assert item.product == "ABI-L1b-RadM"
        assert item.sector == "M2"
        assert item.satellite == "G18"
        assert item.channel == "C01"

    @pytest.mark.parametrize(
        ("name", "product", "sector", "channel"),
        [
            ("OR_ABI-L2-ACMM1-M6_G19", "ABI-L2-ACMM", "M1", None),
            ("OR_ABI-L2-ACHA2KMC-M6_G19", "ABI-L2-ACHA2KMC", None, None),
            ("OR_ABI-L2-MCMIPF-M6_G18", "ABI-L2-MCMIPF", None, None),
            ("OR_ABI-L2-CMIPM2-M6C13_G19", "ABI-L2-CMIPM", "M2", "C13"),
        ],
    )
    def test_parse_l2_keys(self, name, product, sector, channel) -> None:
        stamps = "_s20262801800286_e20262801800343_c20262801800445.nc"
        item = aws.parse_key(name + stamps)
        assert (item.product, item.sector, item.channel) == (product, sector, channel)

    def test_non_abi_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an ABI file name"):
            aws.parse_key("index.html")

    @pytest.mark.parametrize("satellite", ["G19", "g19", 19, "GOES-19", "goes19"])
    def test_bucket_accepts_satellite_spellings(self, satellite) -> None:
        assert aws.bucket(satellite) == "noaa-goes19"

    @pytest.mark.parametrize("satellite", ["G15", 20, "H09", ""])
    def test_bucket_rejects_other_satellites(self, satellite) -> None:
        with pytest.raises(ValueError, match="satellite must be one of"):
            aws.bucket(satellite)


class TestListFiles:
    def test_filters_by_window_and_channel_across_hours(self, fake_s3) -> None:
        h12 = [_key(12, 1, 1), _key(12, 1, 13), _key(12, 56, 13)]
        h13 = [_key(13, 1, 13), _key(13, 31, 13)]
        fake = fake_s3([*h12, *h13, "ABI-L1b-RadC/2026/280/12/junk.txt"])
        files = aws.list_files(
            satellite="G19",
            start=datetime(2026, 10, 7, 12, 30),
            end=datetime(2026, 10, 7, 13, 30),
            channel=13,
        )
        assert [(f.start.hour, f.start.minute) for f in files] == [(12, 56), (13, 1)]
        assert all(f.channel == "C13" for f in files)
        assert files[0].size == 100
        assert fake.calls == [
            "s3://noaa-goes19/ABI-L1b-RadC/2026/280/12/",
            "s3://noaa-goes19/ABI-L1b-RadC/2026/280/13/",
        ]

    def test_channel_filter_skips_l2_files(self, fake_s3) -> None:
        prefix = "ABI-L2-ACMC/2026/280/12/"
        key = (
            f"{prefix}OR_ABI-L2-ACMC-M6_G19_s20262801201178_e20262801203551"
            "_c20262801203578.nc"
        )
        fake_s3([key])
        start = datetime(2026, 10, 7, 12)
        assert aws.list_files(satellite="G19", product="ABI-L2-ACMC", start=start)
        assert not aws.list_files(
            satellite="G19", product="ABI-L2-ACMC", start=start, channel=13
        )

    def test_mesoscale_sector_filter(self, fake_s3) -> None:
        fake_s3([_key(12, 1, 2, product="RadM1"), _key(12, 2, 2, product="RadM2")])
        files = aws.list_files(
            satellite="G19",
            product="ABI-L1b-RadM",
            start=datetime(2026, 10, 7, 12),
            sector="M2",
        )
        assert [f.sector for f in files] == ["M2"]

    def test_aware_datetimes_are_converted_to_utc(self, fake_s3) -> None:
        fake = fake_s3()
        eastern = timezone(timedelta(hours=-4))
        aws.list_files(satellite="G19", start=datetime(2026, 10, 7, 8, tzinfo=eastern))
        assert fake.calls == ["s3://noaa-goes19/ABI-L1b-RadC/2026/280/12/"]

    def test_empty_window_is_rejected(self) -> None:
        t = datetime(2026, 10, 7, 12)
        with pytest.raises(ValueError, match="must be after"):
            aws.list_files(satellite="G19", start=t, end=t)

    @pytest.mark.parametrize("channel", [0, 17, "C99", "red"])
    def test_bad_channel_is_rejected(self, channel) -> None:
        with pytest.raises(ValueError, match="channel must be 1-16"):
            aws.list_files(satellite="G19", start=datetime(2026, 1, 1), channel=channel)


class TestDownload:
    def test_downloads_into_the_directory(self, fake_s3, tmp_path) -> None:
        key = f"ABI-L1b-RadC/2026/280/12/{RADC}"
        fake = fake_s3({key: b"netcdf-bytes"})
        path = aws.download(aws.parse_key(key), tmp_path / "goes")
        assert path == tmp_path / "goes" / RADC
        assert path.read_bytes() == b"netcdf-bytes"
        assert not list((tmp_path / "goes").glob("*.part"))
        assert fake.calls == [f"s3://noaa-goes19/{key}"]

    def test_existing_file_is_reused_unless_overwrite(self, fake_s3, tmp_path) -> None:
        key = f"ABI-L1b-RadC/2026/280/12/{RADC}"
        fake = fake_s3({key: b"new"})
        (tmp_path / RADC).write_bytes(b"old")
        assert aws.download(key, tmp_path).read_bytes() == b"old"
        assert fake.calls == []
        assert aws.download(key, tmp_path, overwrite=True).read_bytes() == b"new"

    def test_missing_object_leaves_nothing(self, fake_s3, tmp_path) -> None:
        fake_s3()
        with pytest.raises(FileNotFoundError):
            aws.download(f"ABI-L1b-RadC/2026/280/12/{RADC}", tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestRetries:
    @pytest.fixture
    def flaky(self, monkeypatch):
        """Replace ``geocloud.files.download`` / ``ls`` with scripted outcomes."""
        files = pytest.importorskip("geocloud.files")

        def install(outcomes: list[Exception | object]) -> list[str]:
            calls: list[str] = []

            def step(uri: str, *args: object, **kwargs: object) -> object:
                calls.append(uri)
                outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            monkeypatch.setattr(files, "download", step)
            monkeypatch.setattr(files, "ls", step)
            return calls

        return install

    def test_spurious_no_such_bucket_is_retried(self, flaky, tmp_path) -> None:
        target = tmp_path / RADC
        calls = flaky([OSError("<Code>NoSuchBucket</Code>"), []])
        assert aws.list_files(satellite="G19", start=datetime(2026, 10, 7, 12)) == []
        assert len(calls) == 2
        # A download's 404 has no body: the directory listing probe tells a
        # spurious NoSuchBucket (retry) from a missing key (stop).
        calls = flaky(
            [
                FileNotFoundError(""),
                OSError("<Code>NoSuchBucket</Code>"),
                target,
            ]
        )
        assert aws.download(RADC, tmp_path) == target
        assert len(calls) == 3  # download, probe listing, download

    def test_missing_key_fails_at_once(self, flaky, tmp_path) -> None:
        calls = flaky([FileNotFoundError(""), []])
        with pytest.raises(FileNotFoundError):
            aws.download(RADC, tmp_path)
        assert len(calls) == 2  # the download and one probe listing

    def test_retries_are_bounded(self, flaky) -> None:
        calls = flaky([OSError("NoSuchBucket")])
        with pytest.raises(OSError, match="NoSuchBucket"):
            aws.list_files(satellite="G19", start=datetime(2026, 10, 7, 12))
        assert len(calls) == aws._ATTEMPTS
