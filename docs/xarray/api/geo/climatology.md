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

::: xrops.anomaly.operators.CalculateClimatology

::: xrops.anomaly.operators.CalculateClimatologySmoothed

::: xrops.anomaly.operators.RemoveClimatology

::: xrops.anomaly.operators.AddClimatology

::: xrops.anomaly.operators.RemoveMean

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.anomaly.calculate_climatology

::: xrops.anomaly.calculate_climatology_season

::: xrops.anomaly.calculate_climatology_smoothed

::: xrops.anomaly.calculate_anomaly

::: xrops.anomaly.calculate_anomaly_smoothed

::: xrops.anomaly.remove_climatology

::: xrops.anomaly.add_climatology

::: xrops.anomaly.remove_mean
