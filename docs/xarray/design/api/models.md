---
status: draft
version: 0.1.0
---

# Models — Layer 2 Graph API

## `core` — Graph API Components

```python
class Node:            # Symbolic intermediate result (see architecture.md §Graph API)
class Input(Node):     # Named graph entry point
    def __init__(self, name: str): ...
class Graph(Operator): # Compiled computation DAG (see architecture.md §Graph API)
    def __init__(self, inputs: dict[str, Input], outputs: dict[str, Node]): ...
```

The Graph API is the Layer 2 high-level user API for building arbitrary computation DAGs. See [architecture.md](../architecture.md#the-functional-graph-api-layer-2) for the full specification of Node, Input, and Graph classes, including use cases for multi-input metrics, branching, and merging workflows.
