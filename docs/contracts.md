# ArchIDE — API & Registry Contracts

This document contains the stable API boundaries and JSON schemas for communication between the ArchIDE frontend and backend.

## 1. Block Registry Specification

All blocks are defined as Python classes in `backend/blocks/` inheriting from `Block` and decorated with `@register_block(category="...")`.

### Block Definition Properties
- **id**: Internal identifier (e.g., `batchnorm2d`).
- **name**: Display name (e.g., `BatchNorm2D`).
- **category**: Sidebar group (e.g., `Normalization`).
- **color**: Node header hex color.
- **is_functional**: If `True`, logic only goes in `forward()`. If `False`, it initializes state in `__init__()`.

### Port & Parameter Definitions
- **inputs**: Array of `PortDef` dictating incoming edges.
- **outputs**: Array of `PortDef` dictating outgoing edges. `var_hint` controls the generated variable name.
- **params**: Array of `ParamDef` defining properties visible in the Inspector. Options: `int`, `float`, `string`, `bool`, `select`, `shape`, `int_tuple`. Supports `auto_infer: bool`.

## 2. Compiler Payload (Graph IR)

The frontend sends the Graph IR to `POST /api/compile`.

```json
{
  "graphs": {
    "main": {
      "name": "main",
      "nodes": [
        {
          "id": "node_abc",
          "type": "custom",
          "data": {
            "block_id": "conv2d",
            "paramValues": { "in_channels": 3, "out_channels": 64 }
          }
        }
      ],
      "edges": [
        { "source": "node_input", "target": "node_abc" }
      ],
      "variables": [
        { "id": "var_1", "name": "CHANNELS", "type": "int", "scope": "local_const", "default": 64 }
      ]
    }
  }
}
```

The backend responds with the multi-file generated `.py` contents, inferred shapes, and error arrays.
