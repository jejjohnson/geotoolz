"""Opt-in benchmark: ``build_raster_catalog`` sequential vs async over S3.

Reads N Sentinel-2 L2A COGs from the public AWS Open Data bucket
``s3://sentinel-cogs/sentinel-s2-l2a-cogs/`` (anonymous, no auth) and
compares the wall-clock time of ``concurrency="sequential"`` against
``concurrency="async"``.

Not registered with pytest-benchmark / CI because it depends on WAN
latency to S3 — numbers are noisy on shared runners. Run locally:

    uv sync --extra fsspec
    uv run python tests/bench/bench_async_build.py

Expected (typical home WAN, ~80 ms RTT, max_concurrent=8):

- sequential: ~30 s for 30 files
- async:      ~5-8 s   for 30 files  (~4-6x speedup)

Tweak ``N_FILES`` and ``MAX_CONCURRENT`` at the bottom to explore the
knee. Anonymous access is enabled via ``storage_options``; no AWS
credentials are required.
"""

from __future__ import annotations

import time

from geocatalog import build_raster_catalog


# Public Sentinel-2 L2A COGs (no auth). One scene per file; 30 scenes
# from a single MGRS tile on different dates.
SCENE_URIS: list[str] = [
    f"s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/T/CH/2023/{m}/S2B_13TCH_2023{m:02d}01_0_L2A/B04.tif"
    for m in range(1, 13)
] + [
    f"s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/T/CH/2023/{m}/S2B_13TCH_2023{m:02d}15_0_L2A/B04.tif"
    for m in range(1, 13)
]

REGEX = r"S2B_13TCH_(?P<date>\d{8})_0_L2A"
STORAGE_OPTIONS = {"anon": True}


def _time_build(*, concurrency: str, max_concurrent: int = 8) -> float:
    t0 = time.perf_counter()
    build_raster_catalog(
        SCENE_URIS,
        filename_regex=REGEX,
        concurrency=concurrency,
        max_concurrent=max_concurrent,
        storage_options=STORAGE_OPTIONS,
    )
    return time.perf_counter() - t0


def main(n_files: int = 24, max_concurrent: int = 8) -> None:
    uris = SCENE_URIS[:n_files]
    print(f"build_raster_catalog: {len(uris)} S2 COGs from s3://sentinel-cogs/")
    print(f"max_concurrent={max_concurrent}\n")

    print("Warming up (one extra run to populate any region/credential caches)…")
    _time_build(concurrency="sequential")

    runs = 3
    for label in ("sequential", "async"):
        times = [
            _time_build(concurrency=label, max_concurrent=max_concurrent)
            for _ in range(runs)
        ]
        mean = sum(times) / len(times)
        spread = max(times) - min(times)
        print(f"  {label:<12}  mean={mean:6.2f}s  spread={spread:6.2f}s  runs={runs}")


if __name__ == "__main__":
    main()
