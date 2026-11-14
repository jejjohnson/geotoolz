"""CF conventions — coordinate validation, time decoding, and renaming.

Layer-1 ``Operator`` wrappers live in :mod:`xrops.cf.operators`.
"""

from __future__ import annotations

from xrops.cf._src.validation import (
    check_dataset_coords,
    decode_cf_time,
    rename_coords,
    rename_from_cf_standard_names,
    rename_to_cf_standard_names,
    rename_variables,
    validate_latitude,
    validate_longitude,
    validate_time,
)
from xrops.cf.operators import (
    DecodeCFTime,
    RenameCoords,
    RenameFromCFStandardNames,
    RenameToCFStandardNames,
    RenameVariables,
    ValidateCoords,
    ValidateLatitude,
    ValidateLongitude,
    ValidateTime,
)


__all__ = [
    "DecodeCFTime",
    "RenameCoords",
    "RenameFromCFStandardNames",
    "RenameToCFStandardNames",
    "RenameVariables",
    "ValidateCoords",
    "ValidateLatitude",
    "ValidateLongitude",
    "ValidateTime",
    "check_dataset_coords",
    "decode_cf_time",
    "rename_coords",
    "rename_from_cf_standard_names",
    "rename_to_cf_standard_names",
    "rename_variables",
    "validate_latitude",
    "validate_longitude",
    "validate_time",
]
