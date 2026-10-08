"""The shared reader toolkit in ``geoproducts._src``: net, files, query, credentials."""

from __future__ import annotations

import base64
import io
import json
import stat
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

from geoproducts._src import credentials, files, net, query, s3
from geoproducts._src.extras import require


# -- retry policy -------------------------------------------------------------


class TestRetryAfter:
    def test_delta_seconds(self) -> None:
        assert net.retry_after_seconds("2.5", attempt=3) == 2.5

    def test_http_date(self) -> None:
        when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
        assert 25 <= net.retry_after_seconds(when, attempt=0) <= 30

    def test_missing_or_garbage_header_backs_off(self) -> None:
        assert net.retry_after_seconds(None, attempt=2, base_s=5.0) == 20.0
        assert net.retry_after_seconds("soon", attempt=0, base_s=5.0) == 5.0

    def test_waits_are_clamped(self) -> None:
        assert net.retry_after_seconds("9999", attempt=0, cap_s=60.0) == 60.0
        assert net.retry_after_seconds("-5", attempt=0) == 0.0
        assert net.backoff_seconds(30, base_s=1.0, cap_s=10.0) == 10.0


class TestRetrying:
    def test_returns_after_transient_exceptions(self) -> None:
        outcomes = iter([OSError("reset"), OSError("reset"), "ok"])
        sleeps: list[float] = []

        def call() -> str:
            value = next(outcomes)
            if isinstance(value, Exception):
                raise value
            return value

        result = net.retrying(
            call,
            wait=lambda o, a: float(a + 1) if isinstance(o, OSError) else None,
            attempts=5,
            sleep=sleeps.append,
        )
        assert result == "ok"
        assert sleeps == [1.0, 2.0]

    def test_returned_failures_are_retried_and_the_last_one_kept(self) -> None:
        calls: list[int] = []
        seen: list[tuple[int, int, float]] = []

        def call() -> int:
            calls.append(1)
            return 429

        out = net.retrying(
            call,
            wait=lambda o, a: 0.0 if o == 429 else None,
            attempts=3,
            sleep=lambda s: None,
            on_retry=lambda o, a, s: seen.append((o, a, s)),
        )
        assert out == 429
        assert len(calls) == 3
        assert seen == [(429, 0, 0.0), (429, 1, 0.0)]

    def test_non_retryable_exception_raises_at_once(self) -> None:
        calls: list[int] = []

        def call() -> None:
            calls.append(1)
            raise ValueError("bad request")

        with pytest.raises(ValueError):
            net.retrying(call, wait=lambda o, a: None, attempts=4)
        assert len(calls) == 1

    def test_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            net.retrying(lambda: None, wait=lambda o, a: None, attempts=0)


def _http_error(code: int, body: bytes = b"", headers=None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(body))  # type: ignore[arg-type]


class TestUrllibWait:
    def test_rate_limit_honours_retry_after(self) -> None:
        assert net.urllib_wait(_http_error(429, headers={"Retry-After": "7"}), 0) == 7.0

    def test_server_errors_and_dropped_connections_back_off(self) -> None:
        assert net.urllib_wait(_http_error(503), 1, base_s=2.0) == 4.0
        assert net.urllib_wait(urllib.error.URLError("timed out"), 0) == 1.0
        assert net.urllib_wait(TimeoutError(), 0) == 1.0

    def test_client_errors_and_values_stop(self) -> None:
        assert net.urllib_wait(_http_error(403), 0) is None
        assert net.urllib_wait(_http_error(404), 0) is None
        assert net.urllib_wait(b"payload", 0) is None

    def test_s3_retries_only_the_spurious_no_such_bucket(self) -> None:
        assert s3.s3_wait(_http_error(404, b"<Code>NoSuchBucket</Code>"), 0) == 1.0
        assert s3.s3_wait(_http_error(404, b"<Code>NoSuchKey</Code>"), 0) is None
        assert s3.s3_wait(_http_error(500), 0) == 1.0


# -- downloads ----------------------------------------------------------------


def test_stream_to_file_is_atomic(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "asset.tif"
    assert net.stream_to_file([b"ab", b"cd"], dest) == dest
    assert dest.read_bytes() == b"abcd"

    def broken():
        yield b"partial"
        raise ConnectionError("reset")

    with pytest.raises(ConnectionError):
        net.stream_to_file(broken(), tmp_path / "other.tif")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sub"]


def test_download_url_retries_then_writes(tmp_path, monkeypatch) -> None:
    failures = [_http_error(503), urllib.error.URLError("reset")]

    def fake_urlopen(url, timeout=None):
        if failures:
            raise failures.pop(0)
        return io.BytesIO(b"x" * 3_000_000)  # several 1 MiB reads

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = net.download_url("https://h/x", tmp_path / "x.bin", sleep=lambda s: None)
    assert dest.stat().st_size == 3_000_000
    assert [p.name for p in tmp_path.iterdir()] == ["x.bin"]


def test_bearer_header_is_scoped_to_one_https_host() -> None:
    host = "api.example.org"
    assert net.bearer_headers_for(f"https://{host}/a", "t", host=host) == {
        "Authorization": "Bearer t"
    }
    assert net.bearer_headers_for(f"http://{host}/a", "t", host=host) == {}
    assert net.bearer_headers_for("https://cdn.example.org/a", "t", host=host) == {}
    assert net.bearer_headers_for(f"https://{host}/a", None, host=host) == {}


def test_list_objects_pages_through_the_bucket(monkeypatch) -> None:
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    pages = {
        None: f'<ListBucketResult xmlns="{ns}"><Contents><Key>a</Key><Size>1</Size>'
        "</Contents><IsTruncated>true</IsTruncated>"
        "<NextContinuationToken>T</NextContinuationToken></ListBucketResult>",
        "T": f'<ListBucketResult xmlns="{ns}"><Contents><Key>b</Key><Size>2</Size>'
        "</Contents><IsTruncated>false</IsTruncated></ListBucketResult>",
    }

    def fake_urlopen(url, timeout=None):
        token = "T" if "continuation-token=T" in url else None
        return io.BytesIO(pages[token].encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert s3.list_objects("bucket", "p/") == [("a", 1), ("b", 2)]
    assert s3.object_url("b", "dir/a b.nc") == "https://b.s3.amazonaws.com/dir/a%20b.nc"


# -- files --------------------------------------------------------------------


class TestAtomicPath:
    def test_public_file_appears_only_on_success(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        with files.atomic_path(dest) as tmp:
            assert tmp.name == "out.txt.part"
            tmp.write_text("x")
            assert not dest.exists()
        assert dest.read_text() == "x"

    def test_failure_removes_the_temporary_file(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError), files.atomic_path(tmp_path / "f") as tmp:
            tmp.write_text("half")
            raise RuntimeError("boom")
        assert list(tmp_path.iterdir()) == []

    def test_private_files_are_owner_only(self, tmp_path: Path) -> None:
        dest = files.write_private_json(tmp_path / "auth.json", {"token": "t"})
        assert json.loads(dest.read_text()) == {"token": "t"}
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600
        assert [p.name for p in tmp_path.iterdir()] == ["auth.json"]


# -- query --------------------------------------------------------------------


class TestQuery:
    def test_valid_bbox(self) -> None:
        query.validate_lonlat_bbox((-104.5, 31.5, -103.5, 32.5))

    @pytest.mark.parametrize(
        ("bbox", "match"),
        [
            ((0, 0, 1), "must be"),
            ((0, -91, 1, 0), "latitudes"),
            ((0, 2, 1, 1), "south > north"),
            ((10, 0, -10, 1), "antimeridian"),
        ],
    )
    def test_invalid_bboxes(self, bbox, match) -> None:
        with pytest.raises(ValueError, match=match):
            query.validate_lonlat_bbox(bbox)  # type: ignore[arg-type]

    def test_times_are_utc(self) -> None:
        eastern = timezone(timedelta(hours=-4))
        aware = datetime(2026, 10, 7, 8, tzinfo=eastern)
        assert query.as_utc(aware) == datetime(2026, 10, 7, 12, tzinfo=UTC)
        assert query.rfc3339_utc(aware) == "2026-10-07T12:00:00Z"
        assert query.time_interval(None, aware) == "../2026-10-07T12:00:00Z"
        assert query.time_interval(None, None) is None


# -- credentials and extras -----------------------------------------------------


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


class TestCredentials:
    def test_jwt_expiry(self) -> None:
        assert credentials.jwt_expiry(_jwt({"exp": 1_800_000_000})) == 1.8e9
        assert credentials.jwt_expiry(_jwt({"sub": "x"})) is None
        assert credentials.jwt_expiry("a.!!!.c") is None
        assert credentials.jwt_expiry("opaque-token") is None

    def test_auth_path_follows_home(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert credentials.auth_path("eumetsat") == (
            tmp_path / ".geoproducts" / "auth_eumetsat.json"
        )

    def test_read_json_config_wants_an_object(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text('{"token": "t"}')
        assert credentials.read_json_config(good) == {"token": "t"}
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2]")
        with pytest.raises(ValueError, match="not a JSON object"):
            credentials.read_json_config(bad)


def test_require_names_the_extra() -> None:
    assert require("json", "feature", "extra").__name__ == "json"
    with pytest.raises(ImportError, match=r"thing needs the \[x\] extra"):
        require("no_such_module_anywhere", "thing", "x")
