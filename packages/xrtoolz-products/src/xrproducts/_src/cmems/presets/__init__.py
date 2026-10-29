"""Per-family CMEMS dataset presets.

Each submodule exposes a ``<NAME>_DATASETS: dict[str, DatasetInfo]``
keyed on the ``copernicusmarine`` ``dataset_id``. The parent
``catalog.py`` merges them into a single :data:`CMEMS_DATASETS` dict.
"""
