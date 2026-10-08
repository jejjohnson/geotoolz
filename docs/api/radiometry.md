# Radiometry

DN ↔ radiance ↔ reflectance conversions, sun/sensor geometry, brightness temperature, and simple
atmospheric correction.

- **DN / radiance / reflectance:** `DNToRadiance`, `DNToReflectance`, `RadianceToDN`,
  `RadianceToReflectance`, `ReflectanceToRadiance`
- **Brightness temperature:** `BTFromRadiance`
- **Sun geometry:** `ComputeSZA`, `EarthSunDistanceCorrection`, `IntegratedIrradiance`
- **Atmospheric correction:** `DOS1` (Chavez dark-object subtraction), `SimpleAtmosphericCorrection`
- **Stretches:** `Gamma`, `MinMaxStretch`, `PercentileClip`, `ToFloat32`
- **Spectral response:** `ApplySRF` — Gaussian SRFs integrated on a 1-nm grid via georeader's
  `transform_to_srf`; source wavelengths from the argument or `attrs["wavelengths"]`, optional
  target `band_names`
- **Tier-A primitives:** `dn_to_radiance`, `radiance_to_dn`, `dn_to_reflectance`, `bt_from_radiance`,
  `dos1`, `min_max_normalize`, `percentile_clip`, `gamma_correct`

::: geotoolz.radiometry
