# Radiometry

DN ↔ radiance ↔ reflectance conversions, sun/sensor geometry, brightness temperature, and simple
atmospheric correction.

- **DN / radiance / reflectance:** `DNToRadiance`, `DNToReflectance`, `RadianceToDN`,
  `RadianceToReflectance`, `ReflectanceToRadiance`
- **Brightness temperature:** `BTFromRadiance`
- **Sun geometry:** `ComputeSZA`, `EarthSunDistanceCorrection`, `IntegratedIrradiance`
- **Atmospheric correction:** `DOS1` (Chavez dark-object subtraction), `SimpleAtmosphericCorrection`
- **Stretches:** `Gamma`, `MinMax`, `PercentileClip`, `ToFloat32`
- **Spectral response:** `ApplySRF` (band-integrated transmittance)

::: geotoolz.radiometry
