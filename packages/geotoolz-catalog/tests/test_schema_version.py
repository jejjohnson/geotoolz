"""Tests for the schema-migration framework (#25).

Covers the three cases laid out in the issue:

1. ``v_artifact == v_current`` → load directly (regression guard).
2. ``v_artifact <  v_current`` → forward migrations chain transparently.
3. ``v_artifact >  v_current`` → `CatalogSchemaError` with the artifact +
   reader versions in the message.

Plus the `geocatalog migrate` CLI subcommand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pytest

from geocatalog import (
    SCHEMA_VERSION_CURRENT,
    CatalogSchemaError,
    from_geoparquet,
    migrate_geoparquet,
    to_geoparquet,
)
from geocatalog._cli import app
from geocatalog._src import parquet as parquet_module


def _build_artifact(
    tmp_path: Path,
    factory: Callable[..., Path],
    *,
    schema_version: int,
) -> Path:
    """Write a 1-row catalog at the given `schema_version`."""
    src = factory((500000, 4000000, 510000, 4010000), "20240601")
    from geocatalog import build_raster_catalog

    cat = build_raster_catalog(
        [src],
        filename_regex=r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
    )
    out = tmp_path / "catalog.parquet"
    to_geoparquet(cat, out, schema_version=schema_version)
    return out


def test_v_current_loads_directly(
    tmp_path: Path, utm29_tile_factory: Callable[..., Path]
) -> None:
    """Regression guard: a catalog at the current version loads as-is."""
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )
    cat = from_geoparquet(out)
    assert len(cat) == 1


def test_v_future_raises_clearly(
    tmp_path: Path, utm29_tile_factory: Callable[..., Path]
) -> None:
    """A v999 artifact raises `CatalogSchemaError` referencing both versions."""
    out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=999)
    with pytest.raises(CatalogSchemaError) as info:
        from_geoparquet(out)
    msg = str(info.value)
    assert "999" in msg
    assert f"v{SCHEMA_VERSION_CURRENT}" in msg
    assert "Upgrade" in msg


def test_forward_migration_chains(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v0 artifact is migrated to v1 transparently when v1 is current."""
    out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=0)

    # Pretend the library is at v1 with one registered v0 -> v1 migration
    # that stamps a marker column. The monkeypatch reverts after the test.
    marker_calls = []

    def _v0_to_v1(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        marker_calls.append(len(gdf))
        gdf = gdf.copy()
        gdf["_migrated_from_v0"] = True
        return gdf

    monkeypatch.setattr(parquet_module, "SCHEMA_VERSION_CURRENT", 1)
    monkeypatch.setattr(parquet_module, "_MIGRATIONS", {0: _v0_to_v1})
    cat = from_geoparquet(out)

    assert marker_calls == [1]
    assert "_migrated_from_v0" in cat.gdf.columns
    assert bool(cat.gdf["_migrated_from_v0"].iloc[0])


def test_migrate_cli_round_trips_v_future(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`geocatalog migrate` rejects an artifact newer than the reader."""
    out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=999)
    try:
        result = app(
            ["migrate", str(out)], exit_on_error=False, result_action="return_value"
        )
    except SystemExit as exc:
        result = exc.code
    assert result == 2
    assert "exceeds reader" in capsys.readouterr().err


def test_migrate_cli_noop_on_current(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`geocatalog migrate` is a no-op when the artifact is already current."""
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )
    try:
        result = app(
            ["migrate", str(out)], exit_on_error=False, result_action="return_value"
        )
    except SystemExit as exc:
        result = exc.code
    assert result == 0
    assert "already at v" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Edge cases surfaced in PR #37 review
# ---------------------------------------------------------------------------


def test_legacy_unversioned_treated_as_v0(
    tmp_path: Path, utm29_tile_factory: Callable[..., Path]
) -> None:
    """A parquet missing `_schema_version` opens as v0, not as v_current.

    Pinning the fallback to v0 (rather than `SCHEMA_VERSION_CURRENT`)
    means the next schema bump will trigger a v0 -> v1 migration for
    legacy files instead of silently skipping it.
    """
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )
    # Rewrite the file without the `_schema_version` column to
    # simulate a pre-versioning artifact.
    legacy_path = tmp_path / "legacy.parquet"
    gdf = gpd.read_parquet(out)
    gdf = gdf.drop(columns=["_schema_version"])
    gdf.to_parquet(legacy_path)

    from geocatalog._src.parquet import _LEGACY_UNVERSIONED, _read_schema_version

    assert _read_schema_version(legacy_path) == _LEGACY_UNVERSIONED == 0
    # Reading still works at v0 (no migration needed today).
    cat = from_geoparquet(legacy_path)
    assert len(cat) == 1


def test_null_schema_version_raises(
    tmp_path: Path, utm29_tile_factory: Callable[..., Path]
) -> None:
    """A `_schema_version` column with NaN values raises `CatalogSchemaError`."""
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )
    # Patch the column to NaN.
    nullified = tmp_path / "nullified.parquet"
    gdf = gpd.read_parquet(out)
    gdf["_schema_version"] = float("nan")
    gdf.to_parquet(nullified)

    with pytest.raises(CatalogSchemaError, match="null values"):
        from_geoparquet(nullified)


def test_mixed_version_shards_raise(
    tmp_path: Path, utm29_tile_factory: Callable[..., Path]
) -> None:
    """A file mixing two `_schema_version` values raises with a clear message."""
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )
    mixed = tmp_path / "mixed.parquet"
    gdf = gpd.read_parquet(out)
    # Duplicate the row so we have two; mark one v0 and one v999.
    gdf2 = gpd.GeoDataFrame(gpd.pd.concat([gdf, gdf], ignore_index=True), crs=gdf.crs)
    gdf2["_schema_version"] = [0, 999]
    gdf2.to_parquet(mixed)

    with pytest.raises(CatalogSchemaError, match="mixed `_schema_version`"):
        from_geoparquet(mixed)


# ---------------------------------------------------------------------------
# Synthetic multi-step migration chain
# ---------------------------------------------------------------------------
#
# `_MIGRATIONS` is empty today (current schema is v0), so the *chained*
# path through `_apply_migrations` — and the "missing migration =
# library bug" guard — have never run against a real schema bump. These
# tests inject a synthetic v0 -> v1 -> v2 chain via monkeypatch (never
# mutating the real registry) to prove the machinery end-to-end before
# the first real bump relies on it.


class TestSyntheticMigrationChain:
    """Prove `_apply_migrations` chains multiple synthetic migrations."""

    @staticmethod
    def _install_chain(
        monkeypatch: pytest.MonkeyPatch,
        calls: list[str],
        *,
        include_1_to_2: bool = True,
    ) -> None:
        """Pretend the library is at v2 with synthetic migrations registered.

        Patches `parquet_module.SCHEMA_VERSION_CURRENT` to 2 and swaps in
        a fresh `_MIGRATIONS` dict (the real registry is never mutated;
        monkeypatch restores both after the test). Each migration appends
        to ``calls`` and stamps a breadcrumb column so both order and
        effect are observable in the output artifact.
        """

        def _v0_to_v1(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            calls.append("0->1")
            gdf = gdf.copy()
            gdf["_migrated_0_1"] = True
            return gdf

        def _v1_to_v2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            calls.append("1->2")
            # Order proof: the v0 -> v1 breadcrumb must already exist.
            assert "_migrated_0_1" in gdf.columns
            gdf = gdf.copy()
            gdf["_migrated_1_2"] = True
            return gdf

        migrations = {0: _v0_to_v1}
        if include_1_to_2:
            migrations[1] = _v1_to_v2
        monkeypatch.setattr(parquet_module, "SCHEMA_VERSION_CURRENT", 2)
        monkeypatch.setattr(parquet_module, "_MIGRATIONS", migrations)

    def test_two_step_chain_applies_in_order(
        self,
        tmp_path: Path,
        utm29_tile_factory: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`migrate_geoparquet` chains v0 -> v1 -> v2, output reads back at v2."""
        out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=0)
        calls: list[str] = []
        self._install_chain(monkeypatch, calls)

        v_before = migrate_geoparquet(out, to_version=2)

        assert v_before == 0
        assert calls == ["0->1", "1->2"]
        # The rewritten artifact is stamped at v2 ...
        assert parquet_module._read_schema_version(out) == 2
        # ... and both transforms landed in the persisted rows.
        cat = from_geoparquet(out)
        assert len(cat) == 1
        assert bool(cat.gdf["_migrated_0_1"].iloc[0])
        assert bool(cat.gdf["_migrated_1_2"].iloc[0])
        # Loading the v2 artifact did not re-run any migration.
        assert calls == ["0->1", "1->2"]

    def test_chain_gap_raises_library_bug(
        self,
        tmp_path: Path,
        utm29_tile_factory: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v2 current with only v0 -> v1 registered: the v1 -> v2 gap raises."""
        out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=0)
        calls: list[str] = []
        self._install_chain(monkeypatch, calls, include_1_to_2=False)

        with pytest.raises(CatalogSchemaError) as info:
            migrate_geoparquet(out, to_version=2)
        msg = str(info.value)
        assert "missing migration v1 -> v2" in msg
        assert "library bug" in msg
        assert calls == ["0->1"]  # the registered step ran; the gap stopped it
        # The failure happened before any rewrite — the artifact is intact
        # at v0, not half-migrated.
        assert parquet_module._read_schema_version(out) == 0

    def test_rerunning_migrate_on_v2_artifact_is_noop(
        self,
        tmp_path: Path,
        utm29_tile_factory: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Migrating the chain's own v2 output again changes nothing."""
        out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=0)
        calls: list[str] = []
        self._install_chain(monkeypatch, calls)
        assert migrate_geoparquet(out, to_version=2) == 0
        calls.clear()
        payload = out.read_bytes()

        v_before = migrate_geoparquet(out, to_version=2)

        assert v_before == 2
        assert calls == []  # no migration re-ran
        assert out.read_bytes() == payload  # file was not rewritten

    def test_migrate_cli_runs_synthetic_chain(
        self,
        tmp_path: Path,
        utm29_tile_factory: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`geocatalog migrate --to-version 2` drives the synthetic chain.

        The CLI re-imports `SCHEMA_VERSION_CURRENT` from the top-level
        `geocatalog` namespace — a *separate binding* from the patched
        `parquet_module.SCHEMA_VERSION_CURRENT` — so the target version
        is passed explicitly via ``--to-version`` rather than patching a
        second namespace. `migrate_geoparquet` itself resolves its
        globals from `parquet_module`, so the single patch reaches the
        CLI path.
        """
        out = _build_artifact(tmp_path, utm29_tile_factory, schema_version=0)
        calls: list[str] = []
        self._install_chain(monkeypatch, calls)

        try:
            result = app(
                ["migrate", str(out), "--to-version", "2"],
                exit_on_error=False,
                result_action="return_value",
            )
        except SystemExit as exc:
            result = exc.code

        assert result == 0
        assert "(v0 -> v2)" in capsys.readouterr().out
        assert calls == ["0->1", "1->2"]
        assert parquet_module._read_schema_version(out) == 2
        cat = from_geoparquet(out)
        assert bool(cat.gdf["_migrated_0_1"].iloc[0])
        assert bool(cat.gdf["_migrated_1_2"].iloc[0])


def test_read_schema_version_cheap_for_migrate(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`migrate_geoparquet` shouldn't load the full gdf when already-current.

    Asserts that `gpd.read_parquet` isn't called when the cheap
    `_read_schema_version` probe already shows the file is up to date.
    """
    out = _build_artifact(
        tmp_path, utm29_tile_factory, schema_version=SCHEMA_VERSION_CURRENT
    )

    from geocatalog import migrate_geoparquet
    from geocatalog._src import parquet as parquet_module

    read_calls = []
    original = parquet_module.gpd.read_parquet

    def _tracking_read(*args, **kwargs):
        read_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(parquet_module.gpd, "read_parquet", _tracking_read)

    before = migrate_geoparquet(out, to_version=SCHEMA_VERSION_CURRENT)
    assert before == SCHEMA_VERSION_CURRENT
    # No full-gdf reads — only the column-selective pyarrow probe.
    assert read_calls == []
