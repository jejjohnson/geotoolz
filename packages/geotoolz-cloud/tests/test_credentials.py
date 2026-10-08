"""Tests for `geocloud.credentials` — registry, file loading, GDAL bridge, redaction.

No network: stores are built and inspected through their ``config``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from obstore.store import MemoryStore

from geocloud import credentials, files
from geocloud.store import clear_obstore_pool, get_obstore, mount, unmount


FUTURE = "2099-01-01T00:00:00Z"
ACCOUNT_SAS = f"sv=2022-11-02&ss=b&srt=sco&sp=rl&se={FUTURE}&sig=c2VjcmV0"
CONTAINER_SAS = f"sv=2022-11-02&sr=c&sp=rl&se={FUTURE}&sig=c2VjcmV0"


@pytest.fixture(autouse=True)
def _fresh_pool():
    clear_obstore_pool()
    yield
    clear_obstore_pool()


# --- redact -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            f"https://a.blob.core.windows.net/c/x.tif?{CONTAINER_SAS}",
            "https://a.blob.core.windows.net/c/x.tif?sv=2022-11-02&sr=c&sp=rl"
            f"&se={FUTURE}&sig=REDACTED",
        ),
        (
            "https://b.s3.amazonaws.com/k?X-Amz-Credential=AKIA%2F1&X-Amz-Signature=ff",
            "https://b.s3.amazonaws.com/k?X-Amz-Credential=REDACTED&X-Amz-Signature=REDACTED",
        ),
        ("https://h/x?token=abc&page=2", "https://h/x?token=REDACTED&page=2"),
        ("Authorization: Bearer eyJ.abc", "Authorization: Bearer REDACTED"),
        ("s3://bucket/plain/key.tif", "s3://bucket/plain/key.tif"),
    ],
)
def test_redact(text, expected):
    assert credentials.redact(text) == expected


def test_errors_from_files_are_redacted():
    with pytest.raises(ValueError) as info:
        files.ls(f"https://host/dir/?{CONTAINER_SAS}")
    assert "c2VjcmV0" not in str(info.value) and "sig=REDACTED" in str(info.value)


# --- set / lookup ---------------------------------------------------------


def test_registered_options_reach_the_pool():
    credentials.set_credentials("s3://pub", anonymous=True, region="us-east-1")
    store = get_obstore("s3://pub/a/b.tif")
    assert store.config["skip_signature"] == "true"
    assert store.config["region"] == "us-east-1"
    # An explicit option wins over the registered one.
    other = get_obstore("s3://pub/a.tif", storage_options={"region": "eu-west-1"})
    assert other.config["region"] == "eu-west-1" and other is not store
    assert credentials.credential_roots() == ["s3://pub"]
    credentials.remove_credentials("s3://pub")
    assert "skip_signature" not in get_obstore("s3://pub/a.tif").config
    credentials.remove_credentials("s3://pub")  # no-op


def test_azure_container_wins_over_account():
    credentials.set_credentials("az://acct", account_key="a2V5")
    credentials.set_credentials("az://acct/raw", sas_token=CONTAINER_SAS)
    assert get_obstore("az://acct/raw/x.tif").config.get("sas_key") == CONTAINER_SAS
    assert get_obstore("az://acct/other/x.tif").config["account_key"] == "a2V5"
    assert (
        get_obstore("https://acct.blob.core.windows.net/raw/x").config["sas_key"]
        == CONTAINER_SAS
    )
    assert credentials.credential_roots() == ["az://acct", "az://acct/raw"]


def test_account_root_spellings():
    credentials.set_credentials(
        "https://acct.blob.core.windows.net", account_key="a2V5"
    )
    assert credentials.credential_roots() == ["az://acct"]


def test_http_host_options():
    headers = {"default_headers": {"Authorization": "Bearer t"}}
    credentials.set_credentials("https://data.example.com", client_options=headers)
    assert credentials.credential_roots() == ["https://data.example.com"]
    assert get_obstore("https://data.example.com/a.tif") is not None


@pytest.mark.parametrize(
    ("uri", "kwargs", "match"),
    [
        ("s3://bucket/key.tif", {}, "not the object"),
        ("hf://org/repo/file.bin", {}, "Hugging Face"),
        ("https://example.com", {"anonymous": True}, "already anonymous"),
        ("s3://bucket", {"sas_token": ACCOUNT_SAS}, "Azure roots only"),
        ("s3://bucket", {"bogus": 1}, "invalid options for s3://bucket"),
        ("az://acct", {"sas_token": CONTAINER_SAS}, "container-scoped"),
        ("az://acct/c", {"sas_token": "sv=1&se=2099-01-01"}, "no `sig=`"),
        (
            "az://acct/c",
            {"sas_token": "sv=1&se=2001-01-01T00:00:00Z&sig=x"},
            "expired at 2001",
        ),
        ("az://acct/c", {"sas_token": "sv=1&se=soon&sig=x"}, "unreadable expiry"),
        (
            "az://acct/c",
            {"sas_token": ACCOUNT_SAS, "sas_key": ACCOUNT_SAS},
            "pass the SAS once",
        ),
    ],
)
def test_register_rejects(uri, kwargs, match):
    with pytest.raises(ValueError, match=match):
        credentials.set_credentials(uri, **kwargs)
    assert credentials.credential_roots() == []


def test_sas_spellings_are_checked_and_normalised():
    credentials.set_credentials("az://acct", sas_key="?" + ACCOUNT_SAS)
    assert get_obstore("az://acct/c/x").config["sas_key"] == ACCOUNT_SAS
    with pytest.raises(ValueError, match="expired"):
        credentials.set_credentials(
            "az://acct/c", azure_storage_sas_key="sv=1&se=2001-01-01&sig=x"
        )


# --- credentials files ----------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_load_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RAW_SAS", CONTAINER_SAS)
    path = _write(
        tmp_path / "creds.toml",
        '["s3://noaa-goes19"]\nanonymous = true\nregion = "us-east-1"\n\n'
        '["az://acct/raw"]\nsas_token = "${RAW_SAS}"\n',
    )
    assert credentials.load_credentials(path) == ["s3://noaa-goes19", "az://acct/raw"]
    assert get_obstore("az://acct/raw/x").config["sas_key"] == CONTAINER_SAS


def test_load_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    unset = _write(tmp_path / "a.toml", '["az://acct/raw"]\nsas_token = "${NOPE}"\n')
    with pytest.raises(ValueError, match=r"\$\{NOPE\}"):
        credentials.load_credentials(unset)
    flat = _write(tmp_path / "b.toml", 'region = "x"\n')
    with pytest.raises(ValueError, match="must be a table"):
        credentials.load_credentials(flat)
    with pytest.raises(FileNotFoundError):
        credentials.load_credentials(tmp_path / "missing.toml")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_load_warns_on_a_shared_file(tmp_path):
    path = _write(tmp_path / "c.toml", '["s3://b"]\nanonymous = true\n')
    path.chmod(0o644)
    with pytest.warns(UserWarning, match="chmod 600"):
        credentials.load_credentials(path)


def test_default_file_loads_on_first_use(tmp_path, monkeypatch):
    path = _write(tmp_path / "creds.toml", '["s3://auto"]\nanonymous = true\n')
    monkeypatch.setenv("GEOCLOUD_CREDENTIALS", str(path))
    assert credentials.credentials_path() == path
    assert get_obstore("s3://auto/x").config["skip_signature"] == "true"
    assert credentials.credential_roots() == ["s3://auto"]


def test_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOCLOUD_CREDENTIALS", "")
    assert credentials.credentials_path() is None
    monkeypatch.delenv("GEOCLOUD_CREDENTIALS")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert credentials.credentials_path() == tmp_path / "geocloud" / "credentials.toml"
    assert (
        credentials.load_credentials() == []
    )  # absent default file: nothing, no error


# --- gdal_access ---------------------------------------------------------


def test_gdal_s3_keys_and_endpoint():
    credentials.set_credentials(
        "s3://private",
        aws_access_key_id="AKID",
        aws_secret_access_key="SECRET",
        aws_session_token="TOKEN",
        region="eu-west-1",
        endpoint="http://localhost:9000",
        request_payer=True,
    )
    path, env = credentials.gdal_access("s3://private/a/b.tif")
    assert path == "/vsis3/private/a/b.tif"
    assert env == {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_HTTPS": "NO",
        "AWS_REGION": "eu-west-1",
        "AWS_REQUEST_PAYER": "requester",
        "AWS_S3_ENDPOINT": "localhost:9000",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_SESSION_TOKEN": "TOKEN",
        "AWS_VIRTUAL_HOSTING": "FALSE",
    }


def test_gdal_gcs_anonymous():
    credentials.set_credentials("gs://pub", anonymous=True)
    assert credentials.gdal_access("gs://pub/x.tif") == (
        "/vsigs/pub/x.tif",
        {"GS_NO_SIGN_REQUEST": "YES"},
    )


def test_gdal_azure_container_sas_goes_in_the_url():
    credentials.set_credentials("az://acct/raw", sas_token=CONTAINER_SAS)
    path, env = credentials.gdal_access("az://acct/raw/dir/scene 1.tif")
    assert path == (
        "/vsicurl/https://acct.blob.core.windows.net/raw/dir/scene%201.tif?"
        + CONTAINER_SAS
    )
    assert env == {}


def test_gdal_azure_account_credentials_are_options():
    credentials.set_credentials("az://acct", sas_token=ACCOUNT_SAS)
    assert credentials.gdal_access("az://acct/raw/x.tif") == (
        "/vsiaz/raw/x.tif",
        {"AZURE_STORAGE_ACCOUNT": "acct", "AZURE_STORAGE_SAS_TOKEN": ACCOUNT_SAS},
    )
    credentials.set_credentials("az://acct", account_key="a2V5")
    assert credentials.gdal_access("az://acct/raw/x.tif").options == {
        "AZURE_STORAGE_ACCESS_KEY": "a2V5",
        "AZURE_STORAGE_ACCOUNT": "acct",
    }
    credentials.set_credentials("az://acct", anonymous=True)
    assert credentials.gdal_access("az://acct/raw/x.tif").options == {
        "AZURE_NO_SIGN_REQUEST": "YES",
        "AZURE_STORAGE_ACCOUNT": "acct",
    }


def test_gdal_plain_and_local(tmp_path):
    assert credentials.gdal_access("https://h/x.tif?sig=1") == (
        "/vsicurl/https://h/x.tif?sig=1",
        {},
    )
    assert credentials.gdal_access(str(tmp_path / "x.tif")).path == str(
        tmp_path / "x.tif"
    )
    assert credentials.gdal_access(f"file://{tmp_path}/x.tif").path == str(
        tmp_path / "x.tif"
    )


def test_gdal_refuses_what_it_cannot_express():
    with pytest.raises(ValueError, match="hf://"):
        credentials.gdal_access("hf://org/repo/x.tif")
    mount("s3://mem", MemoryStore())
    try:
        with pytest.raises(ValueError, match="mounted store"):
            credentials.gdal_access("s3://mem/x.tif")
    finally:
        unmount("s3://mem")

    def provider() -> dict[str, str]:
        return {"access_key_id": "a", "secret_access_key": "b", "token": None}

    credentials.set_credentials("s3://prov", credential_provider=provider)
    with pytest.raises(ValueError, match="credential provider"):
        credentials.gdal_access("s3://prov/x.tif")
