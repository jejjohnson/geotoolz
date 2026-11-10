# API Reference — Composition Core

The Operator / Sequential / Graph composition algebra lives in the
carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit)
framework and is re-exported at `geotoolz.*` (and `pipekit.*`) for
convenience. The only geotoolz-specific addition on this page is
`ModelOp`, which lives at `geoops.learn`.

For the model behind these primitives, read the [Concepts](../concepts.md)
page first.

## Base

::: pipekit_ops.Operator
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.ConfigMixin
    options:
      show_root_heading: true

## Linear composition

::: pipekit_ops.Sequential
    options:
      show_root_heading: true
      show_signature_annotations: true

## Graphs

::: pipekit_ops.Input
    options:
      show_root_heading: true

::: pipekit_ops.Node
    options:
      show_root_heading: true

::: pipekit_ops.Graph
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.Fanout
    options:
      show_root_heading: true
      show_signature_annotations: true

## Inference

::: geoops.learn.ModelOp
    options:
      show_root_heading: true
      show_signature_annotations: true

## Observers — identity with side effects

::: pipekit_ops.Tap
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.Snapshot
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.ShapeTrace
    options:
      show_root_heading: true
      show_signature_annotations: true

## Control flow

::: pipekit_ops.Branch
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.Switch
    options:
      show_root_heading: true
      show_signature_annotations: true

## Building blocks

::: pipekit_ops.Identity
    options:
      show_root_heading: true

::: pipekit_ops.Const
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.Lambda
    options:
      show_root_heading: true
      show_signature_annotations: true

::: pipekit_ops.Sink
    options:
      show_root_heading: true
      show_signature_annotations: true
