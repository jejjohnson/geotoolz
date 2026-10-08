"""``geoproducts.goes.aws`` against a fake S3 endpoint (no network)."""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
from datetime import UTC, datetime, timedelta, timezone

import pytest

from geoproducts.goes import aws


RADC = "OR_ABI-L1b-RadC-M6C13_G19_s20262801201178_e20262801203551_c20262801203578.nc"
RADM = "OR_ABI-L1b-RadM2-M6C01_G18_s20262801200556_e20262801201025_c20262801201053.nc"


def _listing(keys: list[str], *, token: str | None = None) -> bytes:
    contents = "".join(
        f"<Contents><Key>{key}</Key><Size>{100 + i}</Size></Contents>"
        for i, key in enumerate(keys)
    )
    tail = (
        f"<IsTruncated>true</IsTruncated>"
        f"<NextContinuationToken>{token}</NextContinuationToken>"
        if token
        else "<IsTruncated>false</IsTruncated>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}{tail}</ListBucketResult>"
    ).encode()


def _key(hour: int, minute: int, channel: int, *, product: str = "RadC") -> str:
    start = f"2026280{hour:02d}{minute:02d}178"
    return (
        f"ABI-L1b-{product[:4]}/2026/280/{hour:02d}/OR_ABI-L1b-{product}-M6"
        f"C{channel:02d}_G19_s{start}_e{start}_c{start}.nc"
    )


class FakeS3:
    """Serves ListObjectsV2 pages and object bodies; records every URL."""

    def __init__(self, pages: dict[str, list[bytes]], bodies: dict[str, bytes]):
        self.pages = pages
        self.bodies = bodies
        self.urls: list[str] = []

    def urlopen(self, url: str, timeout: float | None = None) -> io.BytesIO:
        self.urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        if parsed.path in {"", "/"}:
            query = urllib.parse.parse_qs(parsed.query)
            pages = self.pages.get(query["prefix"][0], [_listing([])])
            index = int(query.get("continuation-token", ["0"])[0])
            return io.BytesIO(pages[index])
        return io.BytesIO(self.bodies[urllib.parse.unquote(parsed.path[1:])])


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aws, "_BACKOFF_S", 0.0)


def _http_error(url: str, code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(body))  # type: ignore[arg-type]


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch):
    def install(pages=None, bodies=None) -> FakeS3:
        fake = FakeS3(pages or {}, bodies or {})
        monkeypatch.setattr(aws.urllib.request, "urlopen", fake.urlopen)
        return fake

    return install


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
        h12 = [_key(12, 1, 1), _key(12, 1, 13), _key(12, 56, 13), "junk.txt"]
        h13 = [_key(13, 1, 13), _key(13, 31, 13)]
        fake = fake_s3(
            pages={
                "ABI-L1b-RadC/2026/280/12/": [_listing(h12)],
                "ABI-L1b-RadC/2026/280/13/": [_listing(h13)],
            }
        )
        files = aws.list_files(
            satellite="G19",
            start=datetime(2026, 10, 7, 12, 30),
            end=datetime(2026, 10, 7, 13, 30),
            channel=13,
        )
        assert [(f.start.hour, f.start.minute) for f in files] == [(12, 56), (13, 1)]
        assert all(f.channel == "C13" for f in files)
        assert files[0].size == 102
        assert len(fake.urls) == 2
        assert all(
            u.startswith("https://noaa-goes19.s3.amazonaws.com/?") for u in fake.urls
        )

    def test_follows_continuation_tokens(self, fake_s3) -> None:
        prefix = "ABI-L1b-RadC/2026/280/12/"
        fake_s3(
            pages={
                prefix: [
                    _listing([_key(12, 1, 1)], token="1"),
                    _listing([_key(12, 6, 1)]),
                ]
            }
        )
        files = aws.list_files(satellite=19, start=datetime(2026, 10, 7, 12))
        assert [f.start.minute for f in files] == [1, 6]

    def test_mesoscale_sector_filter(self, fake_s3) -> None:
        prefix = "ABI-L1b-RadM/2026/280/12/"
        keys = [_key(12, 1, 2, product="RadM1"), _key(12, 2, 2, product="RadM2")]
        fake_s3(pages={prefix: [_listing(keys)]})
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
        assert "prefix=ABI-L1b-RadC%2F2026%2F280%2F12%2F" in fake.urls[0]

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
        fake = fake_s3(bodies={key: b"netcdf-bytes"})
        path = aws.download(aws.parse_key(key), tmp_path / "goes")
        assert path == tmp_path / "goes" / RADC
        assert path.read_bytes() == b"netcdf-bytes"
        assert not list((tmp_path / "goes").glob("*.part"))
        assert fake.urls == [aws.url("G19", key)]

    def test_existing_file_is_reused_unless_overwrite(self, fake_s3, tmp_path) -> None:
        key = f"ABI-L1b-RadC/2026/280/12/{RADC}"
        fake = fake_s3(bodies={key: b"new"})
        (tmp_path / RADC).write_bytes(b"old")
        assert aws.download(key, tmp_path).read_bytes() == b"old"
        assert fake.urls == []
        assert aws.download(key, tmp_path, overwrite=True).read_bytes() == b"new"

    def test_failed_download_leaves_no_partial_file(
        self, monkeypatch, tmp_path
    ) -> None:
        class Broken(io.BytesIO):
            def read(self, *args):
                raise OSError("connection reset")

        monkeypatch.setattr(aws.urllib.request, "urlopen", lambda *a, **k: Broken())
        with pytest.raises(OSError, match="connection reset"):
            aws.download(RADC, tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestRetries:
    def test_transient_errors_are_retried(self, monkeypatch, tmp_path) -> None:
        failures = [
            _http_error("u", 404, b"<Code>NoSuchBucket</Code>"),
            _http_error("u", 503),
            urllib.error.URLError("timed out"),
        ]
        calls: list[str] = []

        def flaky(url, timeout=None):
            calls.append(url)
            if failures:
                raise failures.pop(0)
            return io.BytesIO(b"payload")

        monkeypatch.setattr(aws.urllib.request, "urlopen", flaky)
        assert aws.download(RADC, tmp_path).read_bytes() == b"payload"
        assert len(calls) == 4

    def test_missing_key_fails_at_once(self, monkeypatch, tmp_path) -> None:
        calls: list[str] = []

        def missing(url, timeout=None):
            calls.append(url)
            raise _http_error(url, 404, b"<Code>NoSuchKey</Code>")

        monkeypatch.setattr(aws.urllib.request, "urlopen", missing)
        with pytest.raises(urllib.error.HTTPError):
            aws.download(RADC, tmp_path)
        assert len(calls) == 1
        assert list(tmp_path.iterdir()) == []

    def test_retries_are_bounded(self, monkeypatch) -> None:
        calls: list[str] = []

        def down(url, timeout=None):
            calls.append(url)
            raise _http_error(url, 500)

        monkeypatch.setattr(aws.urllib.request, "urlopen", down)
        with pytest.raises(urllib.error.HTTPError):
            aws.list_files(satellite="G19", start=datetime(2026, 10, 7, 12))
        assert len(calls) == aws._ATTEMPTS
