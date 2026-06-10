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

::: xrtoolz.geo.operators.CalculateClimatology

::: xrtoolz.geo.operators.CalculateClimatologySmoothed

::: xrtoolz.geo.operators.RemoveClimatology

::: xrtoolz.geo.operators.AddClimatology

::: xrtoolz.geo.operators.RemoveMean

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.calculate_climatology

::: xrtoolz.geo.calculate_climatology_season

::: xrtoolz.geo.calculate_climatology_smoothed

::: xrtoolz.geo.calculate_anomaly

::: xrtoolz.geo.calculate_anomaly_smoothed

::: xrtoolz.geo.remove_climatology

::: xrtoolz.geo.add_climatology

::: xrtoolz.geo.remove_mean
