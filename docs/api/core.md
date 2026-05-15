# API Reference — Core

The composition algebra. Importable as `geotoolz.core.*` and re-exported at
`geotoolz.*` for convenience.

For the model behind these primitives, read the [Concepts](../concepts.md)
page first.

## Base

::: geotoolz.core._src.operator.Operator
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.operator.Carrier
    options:
      show_root_heading: true

## Linear composition

::: geotoolz.core._src.sequential.Sequential
    options:
      show_root_heading: true
      show_signature_annotations: true

## Graphs

::: geotoolz.core._src.graph.Input
    options:
      show_root_heading: true

::: geotoolz.core._src.graph.Node
    options:
      show_root_heading: true

::: geotoolz.core._src.graph.Graph
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.composition.Fanout
    options:
      show_root_heading: true
      show_signature_annotations: true

## Inference

::: geotoolz.core._src.model.ModelOp
    options:
      show_root_heading: true
      show_signature_annotations: true

## Observers — identity with side effects

::: geotoolz.core._src.observers.Tap
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.observers.Snapshot
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.observers.ShapeTrace
    options:
      show_root_heading: true
      show_signature_annotations: true

## Control flow

::: geotoolz.core._src.control.Branch
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.control.Switch
    options:
      show_root_heading: true
      show_signature_annotations: true

## Building blocks

::: geotoolz.core._src.building_blocks.Identity
    options:
      show_root_heading: true

::: geotoolz.core._src.building_blocks.Const
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.building_blocks.Lambda
    options:
      show_root_heading: true
      show_signature_annotations: true

::: geotoolz.core._src.building_blocks.Sink
    options:
      show_root_heading: true
      show_signature_annotations: true
