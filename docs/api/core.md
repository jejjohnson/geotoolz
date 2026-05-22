# API Reference — Composition Core

The Operator / Sequential / Graph composition algebra lives in the
carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit)
framework and is re-exported at `geotoolz.*` (and `pipekit.*`) for
convenience. The only geotoolz-specific addition on this page is
`ModelOp`, which lives at `geotoolz.model`.

For the model behind these primitives, read the [Concepts](../concepts.md)
page first.

## Base

::: pipekit.Operator
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.ConfigMixin
    options:
      show_root_heading: true

## Linear composition

::: pipekit.Sequential
    options:
      show_root_heading: true
      show_signature_annotations: true

## Graphs

::: pipekit.Input
    options:
      show_root_heading: true

::: pipekit.Node
    options:
      show_root_heading: true

::: pipekit.Graph
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.Fanout
    options:
      show_root_heading: true
      show_signature_annotations: true

## Inference

::: geotoolz.model.ModelOp
    options:
      show_root_heading: true
      show_signature_annotations: true

## Observers — identity with side effects

::: pipekit.Tap
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.Snapshot
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.ShapeTrace
    options:
      show_root_heading: true
      show_signature_annotations: true

## Control flow

::: pipekit.Branch
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.Switch
    options:
      show_root_heading: true
      show_signature_annotations: true

## Building blocks

::: pipekit.Identity
    options:
      show_root_heading: true

::: pipekit.Const
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.Lambda
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit.Sink
    options:
      show_root_heading: true
      show_signature_annotations: true
