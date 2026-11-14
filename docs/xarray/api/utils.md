# Utilities

Cross-cutting helpers. `pack_dataset` stacks a Dataset's variables along a `variable` dim, so a carrier-agnostic operator (`pipekit_array.einx`, a `KernelOp`) can take them as one `DataArray`; `unpack_dataset` splits them back. scikit-learn on xarray is [`xrops.learn`](learn.md).

::: xrops.utils.pack_dataset

::: xrops.utils.unpack_dataset
