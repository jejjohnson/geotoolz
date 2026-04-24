"""CDS credential resolution including ``.env`` walk-up."""

from __future__ import annotations

import pytest

from xr_toolz.data._src.credentials import load_cds


@pytest.fixture(autouse=True)
def _clear_cds_env(monkeypatch):
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)


def test_load_cds_explicit_args():
    creds = load_cds(url="https://cds.example", key="key-123")
    assert creds is not None
    assert creds.url == "https://cds.example"
    assert creds.key == "key-123"


def test_load_cds_from_env(monkeypatch):
    monkeypatch.setenv("CDSAPI_URL", "https://cds.env")
    monkeypatch.setenv("CDSAPI_KEY", "env-key")
    creds = load_cds()
    assert creds is not None
    assert creds.url == "https://cds.env"
    assert creds.key == "env-key"


def test_load_cds_from_dotenv(tmp_path, monkeypatch):
    # Walk-up lookup: place ``.env`` in tmp_path and change CWD there.
    (tmp_path / ".env").write_text(
        "CDSAPI_URL=https://cds.dotenv\nCDSAPI_KEY=dotenv-key\n"
    )
    # Also make sure no ``.cdsapirc`` from the real $HOME bleeds in.
    monkeypatch.setenv("HOME", str(tmp_path / "_home"))
    monkeypatch.chdir(tmp_path)
    creds = load_cds()
    assert creds is not None
    assert creds.url == "https://cds.dotenv"
    assert creds.key == "dotenv-key"


def test_load_cds_from_cdsapirc(tmp_path, monkeypatch):
    """Fallback to ``~/.cdsapirc`` when env + dotenv miss."""
    home = tmp_path / "h"
    home.mkdir()
    (home / ".cdsapirc").write_text("url: https://cds.rc\nkey: rc-key\n")
    # Move to an empty cwd so no .env is picked up.
    empty = tmp_path / "workdir"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(empty)
    # Explicit path keeps the test hermetic regardless of HOME.
    creds = load_cds(path=home / ".cdsapirc")
    assert creds is not None
    assert creds.url == "https://cds.rc"
    assert creds.key == "rc-key"


def test_load_cds_returns_none_when_nothing_found(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setenv("HOME", str(empty))
    # No env vars (fixture cleared), no .env, no ~/.cdsapirc.
    assert load_cds(path=empty / "missing") is None


def test_env_wins_over_dotenv(tmp_path, monkeypatch):
    """Env vars take precedence over ``.env`` when both are present."""
    (tmp_path / ".env").write_text(
        "CDSAPI_URL=https://cds.dotenv\nCDSAPI_KEY=dotenv-key\n"
    )
    monkeypatch.setenv("CDSAPI_URL", "https://cds.env")
    monkeypatch.setenv("CDSAPI_KEY", "env-key")
    monkeypatch.chdir(tmp_path)
    creds = load_cds()
    assert creds is not None
    assert creds.url == "https://cds.env"
    assert creds.key == "env-key"


def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("CDSAPI_URL", "https://cds.env")
    monkeypatch.setenv("CDSAPI_KEY", "env-key")
    creds = load_cds(url="https://cds.explicit", key="x")
    assert creds is not None
    assert creds.url == "https://cds.explicit"
    assert creds.key == "x"
