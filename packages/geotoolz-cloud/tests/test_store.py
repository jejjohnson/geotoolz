"""Tests for the process-global obstore client pool (`geocloud.store`).

The pool is the single one shared by geotoolz and geocatalog, so these
tests cover every backend: pool keys, LRU, Azure store construction for
the three URI forms, and that signed ``http(s)`` URLs keep their query.
No network: stores are inspected through their ``config`` / ``prefix`` /
``url`` attributes, and the signed-URL read goes to a localhost server.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest


pytest.importorskip("obstore")

from geocloud import store as public
from geocloud._src import store as objstore


@pytest.fixture(autouse=True)
def _isolate_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Empty pool and no endpoint / region env vars for every test."""
    for var in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_S3_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT",
        "AWS_ENDPOINT_URL_S3",
        "AWS_PROFILE",
        "AZURE_ENDPOINT",
        "HF_ENDPOINT",
        "GOOGLE_SERVICE_ENDPOINT",
        "AZURE_STORAGE_ENDPOINT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HOME",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    # Never pick up a real `huggingface-cli login` token.
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "no-token"))
    objstore.clear_obstore_pool()
    yield
    objstore.clear_obstore_pool()


def test_public_module_reexports_the_one_pool():
    assert public.get_obstore is objstore.get_obstore
    assert public.object_key is objstore.object_key
    assert public.clear_obstore_pool is objstore.clear_obstore_pool
    assert public.get_range_bytes is objstore.get_range_bytes
    assert public.set_obstore_pool_maxsize is objstore.set_obstore_pool_maxsize


# --- key construction ----------------------------------------------------


def test_pool_key_s3_basic():
    key = objstore._pool_key("s3://my-bucket/path/to/file.tif")
    assert key[:2] == ("s3", "my-bucket")


def test_pool_key_gs_basic():
    key = objstore._pool_key("gs://my-bucket/path/to/file.tif")
    assert key == ("gcs", "my-bucket", None, None, None, None, (), None)


def test_pool_key_https_basic():
    key = objstore._pool_key("https://example.com/data/file.tif")
    assert key == ("http", "example.com", None, None, None, None, (), "https")


def test_pool_key_separates_http_from_https():
    assert objstore._pool_key("http://example.com/a") != objstore._pool_key(
        "https://example.com/a"
    )


def test_pool_key_includes_aws_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-3")
    assert objstore._pool_key("s3://bucket-a/key")[3] == "eu-west-3"


def test_pool_key_falls_back_to_default_region(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    assert objstore._pool_key("s3://bucket-a/key")[3] == "us-east-1"


def test_pool_key_different_buckets_distinct():
    assert objstore._pool_key("s3://bucket-a/key") != objstore._pool_key(
        "s3://bucket-b/key"
    )


def test_pool_key_storage_options_are_order_independent():
    a = objstore._pool_key(
        "s3://b/k", {"region": "x", "client_options": {"timeout": "5s", "a": 1}}
    )
    b = objstore._pool_key(
        "s3://b/k", {"client_options": {"a": 1, "timeout": "5s"}, "region": "x"}
    )
    assert a == b


# --- pool identity -------------------------------------------------------


def test_get_obstore_returns_same_instance_for_same_key():
    a = objstore.get_obstore("https://example.com/foo")
    b = objstore.get_obstore("https://example.com/bar")  # same host → same key
    assert a is b


def test_get_obstore_different_hosts_get_different_instances():
    a = objstore.get_obstore("https://example.com/foo")
    b = objstore.get_obstore("https://other.example.com/bar")
    assert a is not b


def test_clear_pool_drops_all_entries():
    a = objstore.get_obstore("https://example.com/foo")
    assert objstore._POOL
    objstore.clear_obstore_pool()
    assert not objstore._POOL
    b = objstore.get_obstore("https://example.com/foo")
    assert a is not b  # fresh client after clear


def test_fork_hook_clears_the_pool():
    objstore.get_obstore("https://example.com/foo")
    objstore._clear_after_fork()
    assert not objstore._POOL


def test_storage_options_participate_in_pool_key():
    """Different options → different client, never a silently reused one."""
    a = objstore.get_obstore("https://example.com/foo")
    b = objstore.get_obstore(
        "https://example.com/foo",
        storage_options={"client_options": {"allow_http": True}},
    )
    c = objstore.get_obstore(
        "https://example.com/bar",
        storage_options={"client_options": {"allow_http": True}},
    )
    assert a is not b
    assert b is c
    assert b.client_options is not None
    assert "allow_http" in b.client_options


# --- LRU eviction --------------------------------------------------------


def test_lru_eviction_at_maxsize():
    objstore.set_obstore_pool_maxsize(2)
    try:
        a = objstore.get_obstore("https://host-a.example.com/x")
        objstore.get_obstore("https://host-b.example.com/x")
        objstore.get_obstore("https://host-c.example.com/x")  # evicts a
        assert objstore.get_obstore("https://host-a.example.com/x") is not a
    finally:
        objstore.set_obstore_pool_maxsize(64)


def test_lru_touch_on_access():
    objstore.set_obstore_pool_maxsize(2)
    try:
        a = objstore.get_obstore("https://host-a.example.com/x")
        objstore.get_obstore("https://host-b.example.com/x")
        objstore.get_obstore("https://host-a.example.com/x")  # touch a
        objstore.get_obstore("https://host-c.example.com/x")  # evicts b
        assert objstore.get_obstore("https://host-a.example.com/x") is a
    finally:
        objstore.set_obstore_pool_maxsize(64)


def test_set_maxsize_rejects_zero():
    with pytest.raises(ValueError, match=">= 1"):
        objstore.set_obstore_pool_maxsize(0)


def test_get_obstore_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="unsupported scheme"):
        objstore.get_obstore("ftp://example.com/foo")


# --- Azure ---------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    [
        "az://myaccount/container-a/path/blob.tif",
        "azure://myaccount/container-a/path/blob.tif",
        "abfs://container-a@myaccount.dfs.core.windows.net/path/blob.tif",
        "abfss://container-a@myaccount.dfs.core.windows.net/path/blob.tif",
        "https://myaccount.blob.core.windows.net/container-a/path/blob.tif",
    ],
)
def test_azure_container_and_key(uri: str):
    """Account from the URI, container bound, no prefix, key = blob path."""
    store = objstore.get_obstore(uri)
    assert type(store).__name__ == "AzureStore"
    assert store.config["account_name"] == "myaccount"
    assert store.config["container_name"] == "container-a"
    assert store.prefix is None
    assert objstore.object_key(uri) == "path/blob.tif"


def test_azure_two_containers_same_account_get_distinct_clients():
    a = objstore.get_obstore("az://acct/container-a/p/b.tif")
    b = objstore.get_obstore("az://acct/container-b/q/c.tif")
    assert a is not b
    assert a.config["container_name"] == "container-a"
    assert b.config["container_name"] == "container-b"
    # Two blobs in one container share one prefix-free client.
    assert objstore.get_obstore("az://acct/container-a/other/d.tif") is a
    assert a.prefix is None


def test_azure_uri_forms_share_one_client():
    """All three spellings of one container hit the same pooled client."""
    a = objstore.get_obstore("az://acct/cont/p.tif")
    b = objstore.get_obstore("abfs://cont@acct.dfs.core.windows.net/p.tif")
    c = objstore.get_obstore("https://acct.blob.core.windows.net/cont/p.tif")
    assert a is b is c


@pytest.mark.parametrize(
    "uri",
    [
        "az://acct",
        "az://acct/",
        "abfs://acct/cont/p.tif",  # abfs must use container@account
    ],
)
def test_azure_malformed_uris_raise(uri: str):
    with pytest.raises(ValueError, match=r"Azure|container@account"):
        objstore.get_obstore(uri)


def test_azure_storage_options_conflict_is_an_error():
    with pytest.raises(ValueError, match="conflicts"):
        objstore.get_obstore(
            "az://acct/cont/p.tif", storage_options={"account_name": "other"}
        )
    # Agreeing values are fine.
    store = objstore.get_obstore(
        "az://acct/cont/p.tif", storage_options={"account_name": "acct"}
    )
    assert store.config["account_name"] == "acct"


def test_prefix_in_storage_options_is_rejected():
    with pytest.raises(ValueError, match="prefix"):
        objstore.get_obstore("az://acct/cont/p.tif", storage_options={"prefix": "x"})


# --- http(s) query strings ----------------------------------------------


def test_signed_url_query_lives_in_the_store():
    uri = "https://h.example.com/a/b.tif?X-Amz-Signature=abc&X-Amz-Expires=60"
    store = objstore.get_obstore(uri)
    assert store.url == "https://h.example.com/?X-Amz-Signature=abc&X-Amz-Expires=60"
    assert objstore.object_key(uri) == "a/b.tif"
    # A different signature is a different client; no query → origin client.
    other = objstore.get_obstore("https://h.example.com/a/b.tif?X-Amz-Signature=zz")
    plain = objstore.get_obstore("https://h.example.com/a/b.tif")
    assert other is not store
    assert plain is not store
    assert plain.url == "https://h.example.com/"


def test_azure_sas_url_is_read_as_signed_http():
    uri = "https://acct.blob.core.windows.net/cont/p.tif?sv=2024&sig=abc"
    store = objstore.get_obstore(uri)
    assert type(store).__name__ == "HTTPStore"
    assert store.url.endswith("?sv=2024&sig=abc")
    assert objstore.object_key(uri) == "cont/p.tif"


@pytest.fixture
def signed_server(tmp_path: Path) -> Iterator[str]:
    """Localhost server that serves byte ranges only with ``sig=ok``."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.bin").write_bytes(b"0123456789")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            if "sig=ok" not in query.split("&"):
                self.send_error(403)
                return
            data = (tmp_path / path.lstrip("/")).read_bytes()
            first, last = self.headers["Range"].split("=")[1].split("-")
            chunk = data[int(first) : int(last) + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_signed_url_request_carries_the_query(signed_server: str):
    """The range request reaches the server with the signature attached."""
    options = {"client_options": {"allow_http": True}}
    read = functools.partial(objstore.get_range_bytes, storage_options=options)
    assert asyncio.run(read(f"{signed_server}/a/b.bin?sig=ok", 2, 4)) == b"2345"
    with pytest.raises(Exception, match=r"403|privileges|Forbidden"):
        asyncio.run(read(f"{signed_server}/a/b.bin", 2, 4))


def test_get_range_bytes_with_explicit_store(tmp_path: Path):
    from obstore.store import LocalStore

    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "b.bin").write_bytes(b"abcdefgh")
    store = LocalStore(prefix=str(tmp_path))
    got = asyncio.run(
        objstore.get_range_bytes("az://acct/cont/p/b.bin", 1, 3, store=store)
    )
    assert got == b"bcd"


# --- review follow-ups (#249) ------------------------------------------


@pytest.mark.parametrize("var", ["AWS_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_S3_ENDPOINT"])
def test_each_s3_endpoint_variable_separates_clients(monkeypatch, var):
    monkeypatch.setenv(var, "http://minio-a:9000")
    a = objstore._pool_key("s3://bucket/k")
    monkeypatch.setenv(var, "http://minio-b:9000")
    b = objstore._pool_key("s3://bucket/k")
    assert a != b


def test_aws_profile_is_part_of_the_key(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "dev")
    a = objstore._pool_key("s3://bucket/k")
    monkeypatch.setenv("AWS_PROFILE", "prod")
    assert objstore._pool_key("s3://bucket/k") != a


@pytest.mark.parametrize(
    ("uri", "key"),
    [
        (
            "hf://datasets/org/repo/data/train.parquet",
            "datasets/org/repo/resolve/main/data/train.parquet",
        ),
        (
            "hf://datasets/org/repo@v1.0/train.parquet",
            "datasets/org/repo/resolve/v1.0/train.parquet",
        ),
        ("hf://org/model/weights.bin", "org/model/resolve/main/weights.bin"),
        ("hf://spaces/org/app/a.json", "spaces/org/app/resolve/main/a.json"),
        ("hf://models/org/model/w.bin", "org/model/resolve/main/w.bin"),
    ],
)
def test_hf_uris_resolve_through_the_hub(uri, key):
    assert objstore.object_key(uri) == key
    assert objstore._pool_key(uri)[:2] == ("hf", "huggingface.co")
    assert "hf" in objstore.SUPPORTED_SCHEMES


def test_hf_endpoint_override(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    assert objstore._pool_key("hf://org/model/w.bin")[:2] == (
        "hf",
        "hf-mirror.example",
    )


def test_hf_uri_without_a_path_raises():
    with pytest.raises(ValueError, match="org/repo/path"):
        objstore.object_key("hf://datasets/org/repo")


def test_hf_endpoint_path_prefix_is_kept(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8080/hf/")
    assert objstore._pool_key("hf://org/model/w.bin")[1] == "localhost:8080/hf"
    monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8080/other")
    assert objstore._pool_key("hf://org/model/w.bin")[1] == "localhost:8080/other"


@pytest.mark.parametrize("prefix", ["kernels", "buckets"])
def test_hf_unsupported_repo_types_are_rejected(prefix):
    with pytest.raises(ValueError, match=f"hf://{prefix}/"):
        objstore.object_key(f"hf://{prefix}/org/repo/file.bin")


def test_hf_empty_revision_is_rejected():
    with pytest.raises(ValueError, match="empty revision"):
        objstore.object_key("hf://org/repo@/weights.bin")


def test_hf_token_keys_the_pool_without_leaking(monkeypatch, tmp_path):
    uri = "hf://org/model/w.bin"
    anonymous = objstore._pool_key(uri)
    monkeypatch.setenv("HF_TOKEN", "secret-a")
    with_a = objstore._pool_key(uri)
    monkeypatch.setenv("HF_TOKEN", "secret-b")
    with_b = objstore._pool_key(uri)
    assert len({anonymous, with_a, with_b}) == 3
    assert "secret" not in repr(with_a)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    assert objstore._pool_key(uri) == anonymous


def test_hf_token_falls_back_to_the_saved_login(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("saved\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))
    assert objstore._hf_token() == "saved"
    monkeypatch.delenv("HF_TOKEN_PATH")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert objstore._hf_token() == "saved"
    monkeypatch.setenv("HF_TOKEN", "env")
    assert objstore._hf_token() == "env"


def test_hf_reads_send_the_token_to_the_prefixed_endpoint(monkeypatch):
    pytest.importorskip("obstore")
    seen: dict[str, str | None] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}/hf")
        monkeypatch.setenv("HF_TOKEN", "secret")
        uri = "hf://datasets/org/repo@v1/data.bin"
        store = objstore.get_obstore(
            uri, storage_options={"client_options": {"allow_http": True}}
        )
        assert bytes(store.get(objstore.object_key(uri)).bytes()) == b"ok"
    finally:
        server.shutdown()
    assert seen == {
        "path": "/hf/datasets/org/repo/resolve/v1/data.bin",
        "auth": "Bearer secret",
    }


@pytest.mark.parametrize(
    ("uri", "key"),
    [
        (
            "hf://datasets/org/repo@refs/pr/3/train.json",
            "datasets/org/repo/resolve/refs%2Fpr%2F3/train.json",
        ),
        (
            "hf://org/repo@refs/convert/parquet/default/train/0.parquet",
            "org/repo/resolve/refs%2Fconvert%2Fparquet/default/train/0.parquet",
        ),
        (
            "hf://org/repo@refs%2Fpr%2F3/w.bin",
            "org/repo/resolve/refs%2Fpr%2F3/w.bin",
        ),
        ("hf://org/repo@v1.0/w.bin", "org/repo/resolve/v1.0/w.bin"),
    ],
)
def test_hf_special_refs_are_kept_whole(uri, key):
    assert objstore.object_key(uri) == key


def test_hf_ref_without_a_file_raises():
    with pytest.raises(ValueError, match="org/repo/path"):
        objstore.object_key("hf://org/repo@refs/pr/3")


def test_hf_endpoint_scheme_is_part_of_the_pool_key(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://host/prefix")
    plain = objstore._pool_key("hf://org/model/w.bin")
    monkeypatch.setenv("HF_ENDPOINT", "https://host/prefix")
    assert objstore._pool_key("hf://org/model/w.bin") != plain


@pytest.mark.parametrize("prefix", ["model", "dataset", "space", "kernel", "bucket"])
def test_hf_singular_type_prefixes_are_rejected(prefix):
    with pytest.raises(ValueError, match=f"use the plural `{prefix}s/`"):
        objstore.object_key(f"hf://{prefix}/org/repo/file.bin")


def test_hf_legacy_token_variable(monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy")
    assert objstore._hf_token() == "legacy"
    monkeypatch.setenv("HF_TOKEN", "new")
    assert objstore._hf_token() == "new"


def test_hf_token_paths_expand_user_and_vars(monkeypatch, tmp_path):
    (tmp_path / "token").write_text("saved")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN_PATH", "~/token")
    assert objstore._hf_token() == "saved"
    monkeypatch.setenv("TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN_PATH", "$TOKEN_DIR/token")
    assert objstore._hf_token() == "saved"
    monkeypatch.delenv("HF_TOKEN_PATH")
    monkeypatch.setenv("HF_HOME", "$TOKEN_DIR")
    assert objstore._hf_token() == "saved"


@pytest.mark.parametrize(
    ("uri", "key"),
    [
        ("hf://org/repo/data#1.csv", "org/repo/resolve/main/data#1.csv"),
        ("hf://org/repo/what?.csv", "org/repo/resolve/main/what?.csv"),
    ],
)
def test_hf_filenames_keep_reserved_characters(uri, key):
    assert objstore.object_key(uri) == key


@pytest.mark.parametrize(
    "uri", ["hf://org/repo/a//b.bin", "hf:///org/repo/a.bin", "hf://org/repo/dir/"]
)
def test_hf_empty_segments_are_rejected(uri):
    with pytest.raises(ValueError, match="empty path segment"):
        objstore.object_key(uri)


def test_hf_reserved_characters_are_encoded_on_the_wire(monkeypatch):
    pytest.importorskip("obstore")
    seen: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen["path"] = self.path
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        uri = "hf://org/repo/data#1.csv"
        store = objstore.get_obstore(
            uri, storage_options={"client_options": {"allow_http": True}}
        )
        assert bytes(store.get(objstore.object_key(uri)).bytes()) == b"ok"
    finally:
        server.shutdown()
    assert seen["path"] == "/org/repo/resolve/main/data%231.csv"


# --- mount ---------------------------------------------------------------


def test_mount_serves_every_uri_under_the_root():
    from obstore.store import MemoryStore

    store = MemoryStore()
    public.mount("s3://mounted", store)
    try:
        assert public.get_obstore("s3://mounted/a/b.tif") is store
        assert (
            public.get_obstore(
                "s3://mounted/c.tif", storage_options={"region": "eu-west-1"}
            )
            is store
        )
        assert public.get_obstore("s3://other/a.tif") is not store
        public.clear_obstore_pool()
        assert public.get_obstore("s3://mounted/a.tif") is store
    finally:
        public.unmount("s3://mounted")
    assert public.get_obstore("s3://mounted/a.tif") is not store
    public.unmount("s3://mounted")  # no-op when nothing is mounted


def test_mount_keys_azure_by_container_and_http_by_scheme():
    from obstore.store import MemoryStore

    store = MemoryStore()
    public.mount("az://acct/one", store)
    public.mount("https://example.com", store)
    try:
        assert public.get_obstore("az://acct/one/k") is store
        assert public.get_obstore("az://acct/two/k") is not store
        assert public.get_obstore("https://example.com/a") is store
        assert public.get_obstore("https://example.com/a?sig=x") is store
        assert public.get_obstore("http://example.com/a") is not store
    finally:
        public.unmount("az://acct/one")
        public.unmount("https://example.com")


def test_mount_rejects_an_object_uri():
    from obstore.store import MemoryStore

    with pytest.raises(ValueError, match="names an object"):
        public.mount("s3://bucket/key.tif", MemoryStore())


# --- local paths --------------------------------------------------------------


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("data/a.tif", "data/a.tif"),
        ("/data/a.tif", "/data/a.tif"),
        (Path("/data/a.tif"), "/data/a.tif"),
        ("file:///data/a%20b.tif", "/data/a b.tif"),
        ("file://localhost/data/a.tif", "/data/a.tif"),
        ("file://server/share/a.tif", "//server/share/a.tif"),
        ("C:/data/a.tif", "C:/data/a.tif"),
        ("C:\\data\\a.tif", "C:\\data\\a.tif"),
        ("a:b/c.tif", "a:b/c.tif"),  # no `://`: a path, not a URI
    ],
)
def test_local_path_names_the_file(location, expected):
    path = public.local_path(location)
    assert path is not None
    assert str(path) == str(Path(expected))


@pytest.mark.parametrize(
    "uri", ["s3://b/a.tif", "gs://b/a.tif", "https://h/a.tif", "hf://o/r/a"]
)
def test_local_path_is_none_for_remote_uris(uri):
    assert public.local_path(uri) is None


def test_local_paths_share_one_pooled_local_store(tmp_path: Path):
    from obstore.store import LocalStore

    a = objstore.get_obstore(tmp_path / "a.tif")
    b = objstore.get_obstore(f"file://{tmp_path}/b.tif")
    assert isinstance(a, LocalStore)
    assert a is b
    assert (
        objstore.object_key(tmp_path / "a.tif")
        == (tmp_path / "a.tif").relative_to(tmp_path.anchor).as_posix()
    )


def test_object_key_of_a_local_path_is_absolute_and_folded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = (tmp_path / "b.bin").relative_to(tmp_path.anchor).as_posix()
    assert objstore.object_key("sub/../b.bin") == expected


def test_get_range_bytes_reads_a_local_file(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"0123456789")
    assert asyncio.run(objstore.get_range_bytes(str(path), 2, 4)) == b"2345"
