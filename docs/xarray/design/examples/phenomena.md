---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    The snippets below import from `xr_toolz.phenomena`, `xr_toolz.metrics.object`,
    and other submodules that **do not exist in the current export surface** — the
    current domain-agnostic functionality still lives under `xr_toolz.geo.*`.
    Treat these imports as design-target aliases; until the modules ship, map each
    `xr_toolz.<topic>` path to its equivalent under today's `xr_toolz.geo.<topic>`.

# Phenomena-Based Validation Examples

Phenomena-based validation treats verification as event detection and characterization. It complements field metrics by asking whether predictions reproduce meaningful finite-amplitude ocean features.

---

## Example 1: Marine Heatwave Verification

### User Story

As a marine heatwave researcher, I want to detect MHWs in predicted and reference SST fields and compare detection skill, duration, intensity, and spatial overlap.

### Motivation

A model can produce low SST RMSE while missing the timing, persistence, or spatial extent of extreme warm events. Event verification makes these errors explicit.

### Demo API

```python
from xr_toolz.phenomena import DetectMarineHeatwaves, MatchObjects
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex, IntersectionOverUnion, DurationError, IntensityBias
```

### Demo Example Usage

```python
events_pred = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_climatology,
    percentile=90,
    min_duration=5,
)(ds_pred)

events_ref = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_climatology,
    percentile=90,
    min_duration=5,
)(ds_ref)

matches = MatchObjects(method="iou", threshold=0.2)(events_pred, events_ref)

pod = ProbabilityOfDetection()(matches)
far = FalseAlarmRatio()(matches)
csi = CriticalSuccessIndex()(matches)
iou = IntersectionOverUnion()(matches)
duration = DurationError()(matches)
intensity = IntensityBias(variable="sst")(matches)
```

---

## Example 2: Eddy Verification

### User Story

As an ocean dynamics researcher, I want to detect mesoscale eddies in predicted and reference SSH fields, so that I can evaluate whether the model captures eddy occurrence, size, amplitude, and trajectory.

### Motivation

Eddies are coherent, finite-amplitude structures. A model can achieve favorable average SSH skill while smoothing eddies, displacing them, or shortening their lifetimes.

### Demo API

```python
from xr_toolz.phenomena import DetectEddies, MatchObjects, ObjectProperties
from xr_toolz.metrics.object import CriticalSuccessIndex, IntersectionOverUnion, CentroidDistance, IntensityBias
```

### Demo Example Usage

```python
eddies_pred = DetectEddies(
    ssh_var="ssha",
    method="closed_contour",
    min_radius=25_000,
    min_lifetime="7D",
)(ds_pred)

eddies_ref = DetectEddies(
    ssh_var="ssha",
    method="closed_contour",
    min_radius=25_000,
    min_lifetime="7D",
)(ds_ref)

matches = MatchObjects(method="iou", threshold=0.1)(eddies_pred, eddies_ref)
props_pred = ObjectProperties(variables=["ssha"])(eddies_pred, ds_pred)

scores = {
    "csi": CriticalSuccessIndex()(matches),
    "iou": IntersectionOverUnion()(matches),
    "centroid_error": CentroidDistance(dims=("lat", "lon"))(matches),
    "amplitude_bias": IntensityBias(variable="ssha")(matches),
}
```

---

## Example 3: Generic EventDefinition

### User Story

As a method developer, I want event definitions to be reusable objects, so that prediction and reference fields are thresholded consistently.

### Motivation

Before scoring events, the phenomenon must be defined objectively: threshold, baseline, minimum duration, minimum area, and connectivity. Making that definition explicit reduces ambiguity.

### Demo API

```python
from xr_toolz.phenomena import EventDefinition, DetectAnomalyObjects, MatchObjects
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex
```

### Demo Example Usage

```python
upwelling_def = EventDefinition(
    variable="sst",
    threshold="climatology_p10",
    baseline=sst_climatology,
    min_duration=3,
    min_area=10_000_000,
    connectivity=8,
    anomaly=True,
)

upwelling_pred = DetectAnomalyObjects(upwelling_def)(ds_pred)
upwelling_ref = DetectAnomalyObjects(upwelling_def)(ds_ref)

matches = MatchObjects(method="iou", threshold=0.2)(upwelling_pred, upwelling_ref)

scores = {
    "pod": ProbabilityOfDetection()(matches),
    "far": FalseAlarmRatio()(matches),
    "csi": CriticalSuccessIndex()(matches),
}
```

---

## Example 4: Event Verification Graph

### User Story

As an applied scientist, I want one graph that detects events, matches them, computes skill scores, and produces a diagnostic panel.

### Motivation

Event verification has multiple dependent steps. The Graph API keeps detection, matching, scoring, and visualization reproducible.

### Demo API

```python
from xr_toolz.core import Graph, Input
from xr_toolz.phenomena import DetectMarineHeatwaves, MatchObjects
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex
from xr_toolz.viz.validation import EventVerificationPanel
```

### Demo Example Usage

```python
pred = Input("prediction")
ref = Input("reference")

pred_events = DetectMarineHeatwaves(sst_var="sst", climatology=sst_clim, percentile=90, min_duration=5)(pred)
ref_events = DetectMarineHeatwaves(sst_var="sst", climatology=sst_clim, percentile=90, min_duration=5)(ref)

matches = MatchObjects(method="iou", threshold=0.2)(pred_events, ref_events)

pod = ProbabilityOfDetection()(matches)
far = FalseAlarmRatio()(matches)
csi = CriticalSuccessIndex()(matches)
panel = EventVerificationPanel()(matches)

event_graph = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={"pod": pod, "far": far, "csi": csi, "panel": panel},
)
```
