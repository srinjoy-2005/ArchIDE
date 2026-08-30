---
name: archide-graph-editing
description: Provides the strict JSON schema requirements for agentically generating or modifying `.arch` files.
---

# ArchIDE `.arch` File Editing Guidelines

When generating or modifying `.arch` graph files programmatically (e.g. for Live-Sync VFS editing), you **MUST** adhere to the following schema constraints. If you fail to include these properties, the React Flow canvas will fail to render the nodes or edges properly.

## 1. Node Schema Requirements

Every node in the `nodes` array must have `"type": "custom"`. If this is omitted, React Flow will render an empty default white box.

Additionally, the `data` object inside each node must contain the full `inputs`, `outputs`, and `params` array schemas that correspond to its `block_id`. Without these, the frontend CustomNode component will not render the connecting handles, which in turn causes React Flow to hide any connected edges.

```json
{
  "id": "linear_1",
  "type": "custom", 
  "position": { "x": 100, "y": 100 },
  "data": {
    "block_id": "linear",
    "label": "Linear",
    "is_functional": false,
    
    // REQUIRED: Must include the schema definitions so the UI can render handles & inspectors
    "inputs": [{"id": "in", "name": "Input", "type": "tensor"}],
    "outputs": [{"id": "out", "name": "Output", "type": "tensor"}],
    "params": [
      {
        "name": "in_features",
        "type": "int",
        "default": 128
      }
    ],
    
    // REQUIRED: The actual values the user has configured
    "paramValues": {
      "in_features": 256
    }
  }
}
```

## 2. Edge Schema Requirements

Every edge in the `edges` array must have `"type": "tensor"`. If omitted, the custom animated SVGs for edges won't render.

```json
{
  "id": "edge_1",
  "source": "linear_1",
  "sourceHandle": "out",
  "target": "relu_1",
  "targetHandle": "in",
  "type": "tensor"
}
```

## 3. Best Practices
- If you don't know the exact `inputs`, `outputs`, or `params` for a block, run a quick Python script importing `backend.blocks.get_all_block_defs` to extract the exact schema rather than guessing.
- Custom Sub-Graphs (like `transformer/ffn`) use `"block_id": "custom"` and must also supply a `"custom_module_id"` field indicating the relative path (e.g. `"transformer/ffn"`).
