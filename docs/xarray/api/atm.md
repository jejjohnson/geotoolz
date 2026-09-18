# Atmosphere

Atmospheric physics on CF datasets. The design rule still holds: only
true physics lives here; anything domain-agnostic belongs in
[`geo`](geo/coords.md). Trace-gas (methane) physics sits under
`xrtoolz.atm.gas.ch4`.

## WRF

`open_wrfout` turns a WRF-ARW `wrfout` file into a CF Dataset on
`(time, level, y, x)`: the `Times` char rows (or `XTIME` minutes) become a
`datetime64[s]` axis, `U` / `V` / `W` are destaggered to mass points
(`U` / `V` rotated from grid- to earth-relative when the file carries
`COSALPHA` / `SINALPHA`), and the split base + perturbation fields are
recombined,

$$
p = P + PB,\qquad
T = (T' + 300\,\mathrm{K})\left(\frac{p}{10^5\,\mathrm{Pa}}\right)^{R_d/c_p},\qquad
z_\text{agl} = \tfrac12\left(z_\text{stag}[k] + z_\text{stag}[k+1]\right) - HGT,
\quad z_\text{stag} = \frac{PH + PHB}{g}.
$$

Everything stays lazy, so `chunks={"time": 1}` keeps the file dask backed.

::: xrtoolz.atm
    options:
      members:
        - open_wrfout
        - destagger
        - wrf_time
        - wrf_wind
        - wrf_height_agl
        - wrf_temperature

## Met diagnostics

Wind in the meteorological convention (bearing the wind blows *from*,
degrees clockwise from north),

$$
|V|=\sqrt{u^2+v^2},\qquad
\theta_\text{from}=\left(270^\circ-\tfrac{180}{\pi}\operatorname{atan2}(v,u)\right)\bmod 360^\circ,\qquad
u=-|V|\sin\theta,\; v=-|V|\cos\theta,
$$

the column integral (trapezoid on the coordinate, or $\sum_k f_k\,\Delta z_k$
with cell widths from the coordinate), the hypsometric equation
$z_{k+1}=z_k+\frac{R_d\,\bar T_{v,k}}{g}\ln\frac{p_k}{p_{k+1}}$, and the
bulk-Richardson boundary-layer height
$Ri_b(z)=\frac{g\,(\theta_v-\theta_{v,s})(z-z_s)}{\theta_{v,s}\,(u^2+v^2)}$.

::: xrtoolz.atm
    options:
      members:
        - wind_speed
        - wind_direction
        - wind_components
        - column_integral
        - hypsometric_height
        - pbl_height_bulk_richardson
        - WindSpeed
        - WindDirection
        - WindComponents
        - ColumnIntegral
        - HypsometricHeight
        - PBLHeightBulkRichardson

## Methane columns

Column averaging kernel in the TROPOMI / OCO convention,
$\hat y=\mathbf h^\top\mathbf x_a+\sum_l h_l A_l (x_l-x_{a,l})$, the
hydrostatic dry-air column
$N_\text{dry}=\frac{N_A}{g\,M_\text{dry}}\int_0^{p_s}(1-q)\,dp$, the
mixing-ratio column $\Omega=\frac{N_A}{g\,M_\text{dry}}\int\chi\,(1-q)\,dp$,
and the single-layer column mass to $\Delta\chi$ conversion
$\Delta\chi=\frac{m\,N_A/M_\text{gas}}{n_\text{air}\,L}$ with
$n_\text{air}=p/(k_B T)$.

::: xrtoolz.atm.gas.ch4
    options:
      members:
        - apply_column_averaging_kernel
        - dry_air_column
        - mixing_ratio_to_column
        - column_mass_to_delta_vmr
        - ApplyColumnAveragingKernel
        - DryAirColumn
        - MixingRatioToColumn
