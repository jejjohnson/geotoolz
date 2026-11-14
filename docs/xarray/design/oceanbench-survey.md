# OceanBench → xrtoolz Integration Survey

Survey of two `oceanbench` repos to identify features worth integrating
into `xrtoolz`. The two repos share a name and lineage but have
diverged into essentially unrelated projects:

- **Repo A**: <https://github.com/jejjohnson/oceanbench> (MIT, last
  pushed 2024-02). SSH-mapping benchmark (DC20a/DC21a Gulf Stream
  OSSE/OSE) — Hydra-driven recipes calling into `oceanbench._src.*`
  Layer-0 functions. The `master` branch is sparse; substantive source
  lives on `neurips-paper` (37 files) and `eman-dev` (35 files).
- **Repo B**: <https://github.com/mercator-ocean/oceanbench> (EUPL-1.2,
  active 2026-05). Operational forecast evaluation harness — Mercator
  GLO-class models (GLO12, GLONET, Wenhai, Xihe, GLO36V1) vs GLORYS
  reanalysis, GLO12 analysis, and Class-IV in-situ observations.
  Papermill-executed jupytext templates + tabular RMSD outputs.

Functional intersection between the two is essentially zero — they
share the abstract idea ("evaluate a candidate ocean model against a
reference") but no source code, metrics, or schema convention. We
treat them as two independent surveys.

This document is a working catalogue: walk through it item-by-item to
decide what to port, adapt, or skip. Status column updates as decisions
are made.

A separate full ocean-data-challenges survey lives at
[ocean-data-challenges-survey.md](ocean-data-challenges-survey.md);
items proposed there are not double-counted here.

---

## A. `jejjohnson/oceanbench` (SSH-mapping)

### A.1 Geoprocessing — [`oceanbench/_src/geoprocessing/`](https://github.com/jejjohnson/oceanbench/tree/neurips-paper/oceanbench/_src/geoprocessing)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.1.1 | [`geostrophic.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/geostrophic.py) | `streamfunction`, `geostrophic_velocities`, `kinetic_energy`, `relative_vorticity` (metpy + pint). | Skip — already in `xrtoolz.ocn` |
| A.1.2 | [`gridding.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/gridding.py) | `coord_based_to_grid` (pyinterp `Binning2D` average), `grid_to_regular_grid` (xesmf bilinear), `interp_da` / `grid_to_coord_based` (pyinterp `Grid3D` trivariate space-time onto along-track). The 3-D grid→coord trivariate path overlaps ODC-1.2 (which uses `xr.Dataset.interp`); 2-D coord→grid binning is a different algorithm. | Skip — declined `pyinterp` / `xesmf` deps; ODC-1.2 covers grid→coord via `xr.Dataset.interp` |
| A.1.3 | [`interpolate.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/interpolate.py) | `fillnan_gauss_seidel` (one-shot `pyinterp.fill.gauss_seidel`). Same idea as ODC-2.2 but pyinterp-backed; would add a heavy dep. | Skip — covered by ODC-2.2 |
| A.1.4 | [`spatial.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/spatial.py) | `transform_360_to_180` / `_180_to_360`, `latlon_deg2m` (degree-axes → meter-axes via `metpy.calc.lat_lon_grid_deltas` + cumulative). | Skip — `latlon_deg2m` is a 2-line recipe via existing [`grid_metrics_from_coords`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/calc/_src/grid_metrics.py): `metrics["dx"].cumsum("lon")` |
| A.1.5 | [`subset.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/subset.py) | `where_slice`, `select_variables`. Trivial. | Skip — `xrtoolz.geo.subset` |
| A.1.6 | [`temporal.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/temporal.py) | `time_rescale` (`(t-t0)/dt` → float seconds with pint units). Useful for spectral analysis where `time` needs to be numeric. | Skip — already shipped at [`transforms/_src/encoders/coord_time.py:14`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/transforms/_src/encoders/coord_time.py#L14) with `Operator` wrapper |
| A.1.7 | [`validation.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/geoprocessing/validation.py) | `validate_latlon` (wrap to [-180,180] + CF attrs), `decode_cf_time`, `validate_time`, `validate_ssh`, `check_time_lat_lon`. Light CF-validation. | `validate_latlon` / `validate_ssh` already covered. Net-new (`decode_cf_time`, `validate_time`, `check_dataset_coords`) filed as [#137](https://github.com/jejjohnson/xrtoolz/issues/137) |

### A.2 Metrics — [`oceanbench/_src/metrics/`](https://github.com/jejjohnson/oceanbench/tree/neurips-paper/oceanbench/_src/metrics)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.2.1 | [`power_spectrum.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/metrics/power_spectrum.py) | `psd_spacetime`, `psd_isotropic`, `psd_*_error`, `psd_*_score` (xrft-backed); **`psd_welch`/`_error`/`_score`** (1-D scipy-Welch on flattened along-track segments); `xr_cond_average`. | Skip — covered by ODC-1.3 (segmented PSD score + λx) |
| A.2.2 | [`stats.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/metrics/stats.py) | `rmse_ds`, **`nrmse_ds`** where `nrmse = 1 − rmse/std_ref` (OceanBench-flavor; differs from xrtoolz's `nrmse = 1 − rmse/sqrt(<ref²>)`). | Filed as [#136](https://github.com/jejjohnson/xrtoolz/issues/136) (bundled with `get_dataset_resolution`) |
| A.2.3 | [`utils.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/metrics/utils.py) | `find_intercept_1D` (scipy interp1d), `find_intercept_2D` (matplotlib contour vertices). | Skip — `xrtoolz.metrics.find_intercept_2D` already present |

### A.3 Preprocessing — [`oceanbench/_src/preprocessing/`](https://github.com/jejjohnson/oceanbench/tree/neurips-paper/oceanbench/_src/preprocessing)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.3.1 | [`alongtrack.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/preprocessing/alongtrack.py) | `alongtrack_ssh = sla_unfiltered + mdt − lwe`, `remove_swath_dimension`, **`select_track_segments`** (gap-aware overlapping segments). | Skip — `calculate_ssh_alongtrack` (#135) + ODC-1.3 cover |
| A.3.2 | [`mean.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/preprocessing/mean.py) | `xr_cond_average` (mask + mean over a subset of dims). | Skip — composable with xarray |

### A.4 Datasets / patcher — [`oceanbench/_src/datasets/`](https://github.com/jejjohnson/oceanbench/tree/neurips-paper/oceanbench/_src/datasets)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.4.1 | [`base.py: XRDABatcher`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/datasets/base.py) | Dataset/iterable yielding stride/patch slices of an `xr.DataArray` with `__len__`, `__getitem__`, `get_coords()`, and **`reconstruct(items, dims_labels=None, weight=None)`** that re-stitches equal-shaped patches back with weighted overlap blending. **Killer feature for ML inference on tiles.** Stand-alone evolution lives in [`jejjohnson/xrpatcher`](https://github.com/jejjohnson/xrpatcher) (MIT). | Proposal: [ob-1.1-xrpatcher-integration.md](ob-1.1-xrpatcher-integration.md) |
| A.4.2 | [`utils.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/oceanbench/_src/datasets/utils.py) | Bookkeeping helpers (`get_dims_xrda`, `update_dict_xdims`, `get_xrda_size`, `get_patches_size`, `get_slices`). | Folded into [ob-1.1](ob-1.1-xrpatcher-integration.md) (already in `xrpatcher._src.utils`) |

### A.5 Hydra config recipes — [`config/`](https://github.com/jejjohnson/oceanbench/tree/master/config)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.5.1 | `config/processing/lib.yaml` + components, `config/metrics/components/*.yaml`, `config/leaderboard/{osse_gf,ose_gf}.yaml`, `config/task/osse_gf_*/task.yaml`, `config/plots/lib.yaml` | Hydra `_target_`-keyed Operator-shaped recipes; `pipe` configs are conceptually `Sequential`. Encode the canonical SSH-mapping eval workflow + region/test-split definitions. | Skip — no Hydra dep in xrtoolz; could reproduce as a `docs/recipes/` cookbook page |

### A.6 Notebook plotting — [`notebooks/dev/neurips-paper/utils.py`](https://github.com/jejjohnson/oceanbench/blob/neurips-paper/notebooks/dev/neurips-paper/utils.py)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| A.6.1 | `PlotPSDIsotropic` / `PlotPSDScoreIsotropic` / `PlotPSDSpaceTime` / `PlotPSDSpaceTimeScore` classes | Heavily overlaps `xrtoolz.viz.validation.PSDIsotropicScorePanel` / `PSDSpaceTimeScorePanel`. Net-new: **secondary-axis wavelength↔wavenumber idiom** via `secondary_xaxis(functions=(lambda x: 1/(x+ε), ...))`. | Skip — already shipped at [`viz/validation/_src/psd.py:88` `_wavelength_axis`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/viz/validation/_src/psd.py#L88), used by all four PSD panels |

---

## B. `mercator-ocean/oceanbench` (operational forecast eval)

### B.1 Core — [`oceanbench/core/`](https://github.com/mercator-ocean/oceanbench/tree/main/oceanbench/core)

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| B.1.1 | [`climate_forecast_standard_names.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/climate_forecast_standard_names.py) | `StandardVariable` / `StandardDimension` enums + **`rename_dataset_with_standard_names(ds)`** that renames variables to their declared CF `standard_name` attr. Clean general-purpose CF normalizer. | Proposal: [ob-1.2-cf-standard-name-rename.md](ob-1.2-cf-standard-name-rename.md) |
| B.1.2 | [`dataset_utils.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/dataset_utils.py) | `Variable`/`Dimension`/`DepthLevel` enums; `VARIABLE_METADATA` `(display_name, unit)` for SSH/T/S/u/v/MLD/geostrophic; `DEPTH_BINS_DEFAULT` (`{"0-5m":(0,5), "5-100m", "100-300m", "300-600m"}`). | Skip — `xrtoolz.types.Variable` covers metadata; `DEPTH_BINS_DEFAULT` shipped via OB-1.4 |
| B.1.3 | [`regions.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/regions.py) | `BoundingBox`, `RegionSpec`, `GLOBAL`, `IBI`, `custom_region(...)`, **`subset_dataset_to_region(ds, region)`** (antimeridian-aware longitude masking; 0–360 vs −180–180; wrap-around when `min > max`), `region_from_dict` / `load_region_file(path)`. Works on both gridded (`isel`) and scattered-point data. | Proposal: [ob-1.3-regions-and-subset.md](ob-1.3-regions-and-subset.md) |
| B.1.4 | [`resolution.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/resolution.py) | `get_dataset_resolution(ds) → "one_degree" \| "quarter_degree" \| "twelfth_degree"`. | Filed as [#136](https://github.com/jejjohnson/xrtoolz/issues/136) (bundled with `nrmse_score`) |
| B.1.5 | [`interpolate.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/interpolate.py) | `interpolate_1_degree(ds)` — `xr.interp` onto a regular 1° grid clipped to bounds. | Skip — composable with `regrid_like` |
| B.1.6 | [`geostrophic_currents.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/geostrophic_currents.py) | dask-array `gradient(ssh)` / Coriolis on sphere; **excludes equator** with `\|lat\| < 0.5° → NaN`. | Skip — already in `xrtoolz.ocn`; equator-mask convention worth noting |
| B.1.7 | [`mixed_layer_depth.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/mixed_layer_depth.py) | `gsw.SA_from_SP` + `gsw.pot_rho_t_exact` → 0.03 kg/m³ density-threshold MLD via `argmax`. | Skip — already in `xrtoolz.ocn`; gsw chain alternative reference |
| B.1.8 | [`lead_day_utils.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/lead_day_utils.py) + [`datetime_utils.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/datetime_utils.py) | `lead_day_labels(1, N) → ["Lead day 1", ...]`, `generate_dates(start, end, delta)`. | Skip — Mercator-specific forecast-cube convention; trivial recipe (`pd.date_range`, list-comp) |
| B.1.9 | [`rmsd.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/rmsd.py) | `rmsd(challenger, reference, variables) → pandas.DataFrame`; rows `"Variable (unit) [key]{depth-label}"`, columns `"Lead day i"`. Spatial (lat,lon) → root → forecast-cycle mean over `first_day_datetime`. | Skip — scoreboard format covered by [OB-1.4 `rmsd_scoreboard`](ob-1.4-class4-profile-validation.md); Mercator forecast-cube driver out of scope |
| B.1.10 | [`metrics.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/metrics.py) | High-level recipes: `rmsd_of_variables_compared_to_{observations, glorys_reanalysis, glo12_analysis}`, MLD/geostrophic/Lagrangian variants. | Skip — recipes, not new computations |
| B.1.11 | [`classIV.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/classIV.py) + [`classIV_support.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/classIV_support.py) | Class-IV validation against in-situ profiles (Argo / drifters / SLA). Pipeline: model SSH→SLA via MDT subtraction → horizontal `interp` model onto `(lat,lon)` per `(first_day, lead_day)` → vertical interp (linear or **monotonic-bracket** for u/v/T/SSH) → RMSD per `(variable, depth_bin, lead_day)` pivoted. | Proposal: [ob-1.4-class4-profile-validation.md](ob-1.4-class4-profile-validation.md) |
| B.1.12 | [`lagrangian_trajectory.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/lagrangian_trajectory.py) + [`lagrangian_support.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/lagrangian_support.py) | OceanParcels `FieldSet`/`ParticleSet`/`JITParticle`/`AdvectionRK4` advection at depth=0; great-circle deviation per lead-day; `lagrangian_particle_count_for_region` scales `N_global=10000` particles by ocean-point ratio (`N_min=2000`). Forecast-cycle weekly chunking + parallel local zarr staging. | Skip — V3 Lagrangian epic ([#49](https://github.com/jejjohnson/xrtoolz/issues/49)) uses scipy backend, not parcels |
| B.1.13 | [`references/{glo12, glorys, observations}.py`](https://github.com/mercator-ocean/oceanbench/tree/main/oceanbench/core/references) | Open canonical Mercator Zarr stores from `https://minio.dive.edito.eu/project-oceanbench/...`; `load_mean_dynamic_topography(resolution)`. | Skip — Mercator-specific data plumbing |
| B.1.14 | [`local_stage.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/local_stage.py) + [`weekly_stage.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/weekly_stage.py) | `should_stage_locally`, `local_stage_directory`, **`open_or_create_local_stage_dataset`**, **`write_dataset_to_local_stage`** (atomic `.tmp → rename`), `run_in_local_stage_workers` (ThreadPoolExecutor), `staged_weekly_dataset(...)`. Production-grade caching. | Skip — xrtoolz isn't growing a persistence layer for now |
| B.1.15 | [`dataset_source.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/dataset_source.py) | Tags Datasets with `oceanbench_source_{kind,name,resolution}` attrs so downstream stages reuse stage directories. | Skip — coupled to B.1.14 staging which is out of scope |
| B.1.16 | [`evaluate.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/evaluate.py) + [`templates/evaluation_template.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/templates/evaluation_template.py) | `oceanbench evaluate --challenger=path.py --region=<id>` jupytext+papermill drives a 12-cell evaluation notebook to a `<stem>.<region>.report.ipynb`. | Skip — papermill / jupytext / Quarto framework |

### B.2 Public API + samples

| # | File / Symbol | What it provides | Status |
|---|------|------------------|--------|
| B.2.1 | [`oceanbench.{metrics, regions, datasets/*}`](https://github.com/mercator-ocean/oceanbench/tree/main/oceanbench) | Thin shims re-exporting `core/`. | Skip — Mercator API surface |
| B.2.2 | [`assets/*.py`](https://github.com/mercator-ocean/oceanbench/tree/main/assets) + [`challenger_datasets/*.py`](https://github.com/mercator-ocean/oceanbench/tree/main/challenger_datasets) | Schema convention: `(first_day_datetime, lead_day_index, [depth,] latitude, longitude)` with vars `{zos, thetao, so, uo, vo}`. Pre-generated `*.global.report.ipynb` reports. | Document the schema convention as a recipe |
| B.2.3 | [`website/`](https://github.com/mercator-ocean/oceanbench/tree/main/website) | Quarto-rendered leaderboard reading scores from papermill output. | Skip — Mercator publication tooling |

---

## Cross-cutting workflows

### Repo A — DC20a/DC21a Gulf Stream evaluation

```text
task config (region, splits, obs-mix)
  → data adapter: open_dataset → rename → validate_latlon/time/ssh → sortby → compute
  → preprocessing: resample 1D → fillnan_gauss_seidel → latlon_deg2m → time_rescale
  → for-each-method:
      → grid metrics:
            psd_isotropic_score(study, ref, "ssh", ["lon","lat"], avg_dims=["time"])
            psd_spacetime_score(study, ref, "ssh", ["time","lon"], avg_dims=["lat"])
            nrmse_ds(stack(ref, study), ...)
            find_intercept_1D / find_intercept_2D → λx, λt
      → alongtrack metrics:
            interp_da(grid → track) → select_track_segments → psd_welch_score → λx
      → plots: PSDIsotropic / PSDSpaceTime / hvplot maps
  → leaderboard table + summary YAML
```

### Repo B — Operational challenger evaluation

```text
challenger_dataset = challenger_loader()    # (first_day, lead_day, depth?, lat, lon) Zarr
  → rename_dataset_with_standard_names(ds)
  → subset_dataset_to_region(ds, region)
  → for-each-reference {observations | glorys | glo12-analysis}:
       → for-each-metric {variables | MLD | geostrophic | Lagrangian}:
            → stage references locally (weekly_stage parallelism)
            → either:
                rmsd(challenger - reference) over (lat, lon, first_day)
                  → pandas pivot (variable, depth_bin, lead_day)
              or:
                parcels.AdvectionRK4 weekly → great-circle deviation by lead-day
              or:
                classIV: interp model→obs (horizontal) → bracket vertical interp
                  → RMSD pivot
  → papermill renders evaluation notebook → publish to website
```

---

## Top integration candidates (ranked by value × ease, **only items not already covered**)

| Rank | Item | Source | Value | Ease |
|---|---|---|---|---|
| 1 | **`XRDABatcher`** — patches/strides + reconstruct with overlap blending for ML inference | A.4.1 | HIGH | MEDIUM |
| 2 | **`rename_with_cf_standard_names`** — CF normalizer | B.1.1 | HIGH | TINY |
| 3 | **`RegionSpec` + antimeridian-aware `subset_to_region`** | B.1.3 | HIGH | SMALL |
| 4 | **Class-IV grid→scattered-obs interp + bracket vertical + (var, depth, lead) RMSD pivot** | B.1.11 | HIGH | MEDIUM |
| 5 | **`nrmse_score`** (1 − rmse/std_ref, OceanBench convention) for DC20a/DC21a parity | A.2.2 | MEDIUM | TINY |
| 6 | **(variable × depth × lead-day) RMSD scoreboard formatter** | B.1.9 | MEDIUM | SMALL |
| 7 | **Local-stage / weekly-stage Zarr primitives** (atomic `.tmp → rename`, ThreadPool) | B.1.14 | MEDIUM-HIGH | MEDIUM |
| 8 | `get_dataset_resolution(ds)` snap-to-known-grid | B.1.4 | LOW | TINY |
| 9 | `time_rescale` + `latlon_deg2m` named ops | A.1.4 / A.1.6 | LOW | TINY |
| 10 | Secondary wavelength↔wavenumber axis idiom (fold into existing PSD panels) | A.6.1 | LOW | TINY |
| 11 | OceanParcels-based Lagrangian advection (alternative reference for V3 epic) | B.1.12 | — | document only |

---

## Items to skip

- All Hydra `_target_` plumbing (Repo A) — `xrtoolz.core.Sequential` /
  `Operator` already cover this; could reproduce DC20a as a
  `docs/recipes/` cookbook page.
- `oceanbench._src.geoprocessing.geostrophic` — already in `xrtoolz.ocn`.
- `select_track_segments` — covered by ODC-1.3 (segmented PSD score + λx).
- `fillnan_gauss_seidel` (pyinterp) — covered by ODC-2.2 (Laplacian
  gap-fill); pyinterp variant adds a heavy dep for marginal gain.
- `psd_spacetime`, `psd_isotropic`, `psd_*_score` — already in
  `xrtoolz.transforms` / `xrtoolz.metrics`; confirm parity by test.
- `find_intercept_2D` — already in `xrtoolz.metrics`.
- Repo B's `geostrophic_currents` and `mixed_layer_depth` — already in
  `xrtoolz.ocn`; gsw-backed MLD chain documented as alternative.
- Mercator's challenger / reference Zarr loaders pointing at
  `minio.dive.edito.eu/project-oceanbench/...` — Mercator-specific.
- Mercator's papermill / jupytext / Quarto / website tooling.
- OceanParcels-based Lagrangian — defer to V3 epic (#49) which has its
  own design; document Parcels as alternative reference only.

---

## License notes

- **Repo A** (`jejjohnson/oceanbench`) — MIT. Compatible.
- **Repo B** (`mercator-ocean/oceanbench`) — EUPL-1.2 (weak copyleft).
  If we *port* code from Repo B verbatim we should pull
  algorithms / ideas only and re-implement under xrtoolz's MIT, with
  attribution in `THIRD_PARTY.md`. Items #1–#7 above are short enough
  to re-implement cleanly without code copying.

---

## Decision log

Use this section as we walk through items A.x and B.x to record
decisions (port / adapt / skip), proposed APIs, and follow-up issue
links. Conventional naming: **OB-X.Y** for proposals analogous to
ODC-X.Y.

| Item | Decision | Notes |
|------|----------|-------|
| _e.g._ A.4.1 | Adapt as `xrtoolz.patcher.XRDABatcher` | OB-1.1 |
