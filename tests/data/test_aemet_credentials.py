"""AEMET credential loader tests (.env + env var + on-disk)."""

from __future__ import annotations

from xr_toolz.data import load_aemet


def test_load_aemet_prefers_explicit_arg(monkeypatch, tmp_path):
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    c = load_aemet(api_key="explicit")
    assert c is not None and c.api_key == "explicit"


def test_load_aemet_reads_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("AEMET_API_KEY", "from-env")
    monkeypatch.chdir(tmp_path)
    c = load_aemet()
    assert c is not None and c.api_key == "from-env"


def test_load_aemet_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    (tmp_path / ".env").write_text("AEMET_API_KEY=from-dotenv\nOTHER=ignored\n")
    monkeypatch.chdir(tmp_path)
    c = load_aemet()
    assert c is not None and c.api_key == "from-dotenv"


def test_load_aemet_dotenv_ignored_if_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("AEMET_API_KEY", "wins")
    (tmp_path / ".env").write_text("AEMET_API_KEY=loses\n")
    monkeypatch.chdir(tmp_path)
    c = load_aemet()
    assert c is not None and c.api_key == "wins"


def test_load_aemet_reads_home_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / ".aemet"
    cfg.write_text("api_key: from-home\n")
    c = load_aemet(path=cfg)
    assert c is not None and c.api_key == "from-home"


def test_load_aemet_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    # No .env, no home-config override, nothing.
    assert load_aemet(path=tmp_path / "nothing") is None
