# Climatology & Anomalies

Compute climatologies and seasonal cycles, then form anomalies by removing
them. For a field $x(t)$ with climatology $\bar{x}(\tau)$ over the cycle
phase $\tau$ (e.g. day-of-year), the anomaly is

$$
x'(t) = x(t) - \bar{x}\bigl(\tau(t)\bigr).
$$

The `*_smoothed` variants apply a harmonic / rolling smoother to
$\bar{x}(\tau)$ before subtraction.

## Operators

::: xrops.geo.operators.CalculateClimatology

::: xrops.geo.operators.CalculateClimatologySmoothed

::: xrops.geo.operators.RemoveClimatology

::: xrops.geo.operators.AddClimatology

::: xrops.geo.operators.RemoveMean

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geo.calculate_climatology

::: xrops.geo.calculate_climatology_season

::: xrops.geo.calculate_climatology_smoothed

::: xrops.geo.calculate_anomaly

::: xrops.geo.calculate_anomaly_smoothed

::: xrops.geo.remove_climatology

::: xrops.geo.add_climatology

::: xrops.geo.remove_mean
