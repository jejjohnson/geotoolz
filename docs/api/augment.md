# `geotoolz.augment` — RS-safe augmentations

Curated augmentation operators for training on remote-sensing `GeoTensor`
patches. Spatial operators preserve CRS and either preserve the existing
transform or update it consistently; radiometric operators treat bands as
physical measurements instead of generic RGB channels.

## Quick pipeline

```python
import geotoolz as gz

augment = gz.augment.Compose(
    [
        gz.augment.RandomFlip(p_horizontal=0.5, p_vertical=0.5),
        gz.augment.RandomRotate90(p=0.75),
        gz.augment.BrightnessJitter(factor=(0.95, 1.05), per_band=True),
        gz.augment.GaussianNoise(sigma=0.01),
    ]
)

augmented_patch = augment(training_patch, seed=0)
```

## Before / after samples

| Family | Before | After |
| --- | --- | --- |
| Geometry | `patch` | `RandomFlip(p_horizontal=1)(patch)` mirrors pixels and updates the affine transform. |
| Geometry | `patch` | `RandomRotate90(p=1)(patch, seed=0)` uses one of `np.rot90(..., k=1..3)` and updates transform orientation. |
| Geometry | `patch` | `RandomCrop(size=(256, 256))(patch, seed=0)` returns a smaller georeferenced chip. |
| Geometry | `patch` | `RandomShift(max_shift=(8, 8))(patch, seed=0)` reads a shifted fixed-size window with padding at edges. |
| Radiometric | `patch` | `BrightnessJitter(per_band=True)(patch, seed=0)` samples one gain per band. |
| Radiometric | `patch` | `ContrastJitter(per_band=False)(patch, seed=0)` applies one contrast factor to every band. |
| Noise | `patch` | `GaussianNoise(sigma=0.01)(patch, seed=0)` adds reproducible additive sensor noise. |
| Noise | `patch` | `SpeckleNoise(sigma=0.05)(patch, seed=0)` applies multiplicative SAR-style noise. |
| Band-level | `patch` | `BandDropout(p=0.1, fill=0)(patch, seed=0)` independently masks bands. |
| Band-level | `patch` | `BandJitter(groups={"swir": ["B11", "B12"]})(patch, seed=0)` only permutes configured band groups. |
| RS-specific | `patch` | `SunAngleJitter(delta_sza_deg=(-3, 3))(patch, seed=0)` rescales reflectance for a simulated SZA shift. |
| RS-specific | `patch` | `AtmosphericHaze(intensity=(0, 0.05))(patch, seed=0)` adds stronger haze to shorter wavelengths. |
| RS-specific | `patch` | `SimulatedClouds(coverage=(0, 0.2))(patch, seed=0)` blends in a smooth synthetic cloud field. |
| Mixing | `patch` | `CutMix(pool=[other_patch], p=0.5)(patch, seed=0)` pastes a rectangle from a same-shaped pool sample. |

## Seeding contract

Every random operator accepts a constructor `seed` and a per-call `seed`
override:

```python
op = gz.augment.GaussianNoise(sigma=0.01, seed=1)
same_a = op(patch, seed=42)
same_b = op(patch, seed=42)
```

Use the same seed for multiple geometric pipelines when different modalities
must receive identical spatial transforms.

## Composer

::: geotoolz.augment._src.operators.Compose
    options:
      show_root_heading: true
      show_signature_annotations: true

## Geometry

::: geotoolz.augment._src.operators.RandomFlip
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.RandomRotate90
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.RandomCrop
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.RandomShift
    options:
      show_root_heading: true
      show_signature_annotations: true

## Radiometric

::: geotoolz.augment._src.operators.BrightnessJitter
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.ContrastJitter
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.GaussianNoise
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.SpeckleNoise
    options:
      show_root_heading: true
      show_signature_annotations: true

## Band-level

::: geotoolz.augment._src.operators.BandDropout
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.BandJitter
    options:
      show_root_heading: true
      show_signature_annotations: true

## RS-specific

::: geotoolz.augment._src.operators.SunAngleJitter
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.AtmosphericHaze
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.SimulatedClouds
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.augment._src.operators.CutMix
    options:
      show_root_heading: true
      show_signature_annotations: true

