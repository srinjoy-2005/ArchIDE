# ArchiDE: Roadmap & Future Features

This document outlines the short-term and long-term goals for the ArchiDE visual builder. 

## 🚧 Short-Term Features

### 1. Block Properties & Hyperparameter Tuning
- **Current State**: We have a Properties Sidebar that correctly syncs with uncontrolled React Flow nodes, allowing users to modify parameters.
- **To Do**: Expand the UI for parameter inputs (sliders, dropdowns for activation functions, etc.) and validate input types (e.g. ensuring `in_features` is an integer > 0).

### 2. Advanced Graph Validation & Shape Inference
- **Current State**: Kahn's algorithm detects cyclic dependencies in the backend. However, users can connect incompatible blocks without dimension validation.
- **To Do**: 
  - Implement a shape propagation engine (potentially running in the Python backend) to calculate tensor dimensions as they flow through the network.
  - Display visual warnings on the React Flow canvas if incompatible blocks are connected (e.g., dimension mismatch).


### 3. Complex Architectures (Non-Sequential)
- **Current State**: The backend compiler supports basic mathematical ops (`Add`, `Subtract`) and some stateful layers.
- **To Do**: Update the parsing logic in `backend/compiler.py` to fully support complex branching architectures, advanced skip connections (ResNets), and multi-input/output blocks (like `torch.cat`).

## 🚀 Long-Term Features

### 1. Save/Load & Export Capabilities
- **To Do**: 
  - Implement serialization so users can save their visual graph as a `.json` file (or a database record) and reload it later.
  - Add a feature to download the generated `.py` file directly to the user's local machine, or export the weights if a training loop is added.

### 2. Expand Block Library
- **To Do**: Gradually add more advanced blocks to `backend/registry.py` (e.g., Transformer Blocks, RNNs/LSTMs, custom loss functions, and optimizers).

### 3. JSON Schema Reference Design
- **Note**: Early on, a TypeScript-based JSON schema logic was proposed (detailed in `graph_ir_spec.md` and old IR preview files). While the actual implementation of the backend logic has been refactored entirely into Python (FastAPI/Pydantic), that original design remains a highly compelling reference. Future iterations involving complex graph serialization or frontend-backend synchronization should look to that design for inspiration.


# TODO

- Fundamental blocks when connected there should be a static check for shape compatibility. ✅ DONE (Shape Inference Pass)
- decoupling of block functionalities, like input shapes output shapes and block core function : DESIGN ✅ DONE (OOP IR nodes in `backend/blocks/`)
- Compiler design: going to use established CD principles to make this scalable to custom blocks, need to think about where the code written for a block is placed and where parameters that will be verified are placed ✅ DONE

---

## ✅ Completed Tasks

### Task A: Unique, Readable Variable Names for Input Nodes ✅ DONE
**`backend/compiler.py`** — `_build_input_var_map()` pre-builds a disambiguated map for all input nodes before code generation. If two input nodes share a label (e.g. both "Input"), they become `x_input`, `x_input_2`, etc. All other references resolve through this map, so no collision is possible.

### Task B: Static Tensor Dimension Checking ✅ DONE
**`backend/compiler.py`** — `shape_inference_pass()` now runs as the **first step** of `generate_pytorch_code()`, before any code is emitted. It propagates concrete shapes through the graph and raises a structured `ShapeError` on mismatch. Dimensions with value `"ANY"` are treated as dynamic and are propagated transparently downstream without blocking compilation.

**`backend/main.py`** — The `/api/compile` endpoint now catches `ShapeError` separately from generic `ValueError` and returns a **HTTP 422** with a structured JSON body:
```json
{
  "error": "ShapeMismatch",
  "message": "Conv2D expected in_channels=3, but got 64",
  "node_id": "node-abc123",
  "node_label": "Conv2D"
}
```

---

## 🖥️ Frontend TODO — Surfacing Shape Errors to the User

The backend now returns a structured `422` with `node_id` and `node_label` when a shape mismatch is detected. The frontend needs to consume this and display it intuitively. Here are the changes needed in `src/app/page.tsx`:

### TODO 1 — Parse the 422 error response in the compile handler
In the `handleCompile` function, after `await fetch(...)`:
```ts
// TODO: frontend/compile-handler
if (!response.ok) {
  const err = await response.json();
  if (err.detail?.error === "ShapeMismatch") {
    // Mark the offending node with an error state
    setShapeErrorNodeId(err.detail.node_id);
    setGeneratedCode(`# ❌ Shape Mismatch at "${err.detail.node_label}":\n# ${err.detail.message}`);
  } else {
    setGeneratedCode(`# ❌ Error: ${err.detail}`);
  }
  return;
}
setShapeErrorNodeId(null); // clear on success
```

### TODO 2 — Add `shapeErrorNodeId` state
```ts
// TODO: frontend/state
const [shapeErrorNodeId, setShapeErrorNodeId] = useState<string | null>(null);
```

### TODO 3 — Highlight the offending node in React Flow
The custom node renderer should read `shapeErrorNodeId` from shared state (or a Zustand store) and apply a red error border if the current node's ID matches:
```tsx
// TODO: frontend/custom-node-style
const isError = shapeErrorNodeId === node.id;
<div className={`node-wrapper ${isError ? "border-2 border-red-500 shadow-red-500/40 shadow-lg" : ""}`}>
```

### TODO 4 — Show an inline error tooltip on the offending node
When `isError` is true, render a small floating badge above the node:
```tsx
// TODO: frontend/error-badge
{isError && (
  <div className="absolute -top-6 left-0 bg-red-600 text-white text-xs px-2 py-0.5 rounded shadow">
    Shape mismatch
  </div>
)}
```

### TODO 5 — Clear error state when the user edits the graph
In the `onNodesChange` / `onEdgesChange` React Flow callbacks, call `setShapeErrorNodeId(null)` so the red highlight disappears as soon as the user makes any change, signalling the error may have been fixed.
```ts
// TODO: frontend/clear-error-on-change
onNodesChange={(changes) => { setShapeErrorNodeId(null); applyNodeChanges(changes, nodes); }}
```

---

## 🖥️ Frontend TODO — User-Overridable Output Variable Names

The backend assigns readable default names to each block's output (`conv_feat`, `sum`, `probs`, etc.) via `var_hint` on `PortDef`. The user should be able to override these names from the Properties Panel to give outputs more project-specific names (e.g. `rgb_features`, `attention_scores`).

### Backend contract
The backend reads `node.data.paramValues["_output_aliases"]` (a dict) when choosing output variable names. A key present in this dict overrides the `var_hint` default:
```python
# In compiler.py — _build_output_var()
aliases = params.get("_output_aliases", {})
if port.id in aliases and aliases[port.id]:
    base = aliases[port.id]   # user-supplied name wins
elif port.var_hint:
    base = port.var_hint      # block default
else:
    base = f"x_{node_id[:8]}"
```

### TODO 6 — Add output alias inputs to the Properties Panel
For each output port of the selected node, render a text input in the Properties Panel:
```tsx
// TODO: frontend/output-alias-input
{selectedBlock?.outputs.map(port => (
  <div key={port.id} className="flex flex-col gap-1">
    <label className="text-xs text-slate-400 font-mono">
      Output: <span className="text-slate-200">{port.name}</span>
    </label>
    <input
      type="text"
      placeholder={port.var_hint ?? "variable name"}
      value={node.data.paramValues?._output_aliases?.[port.id] ?? ""}
      onChange={(e) => updateNodeParam(node.id, `_output_aliases.${port.id}`, e.target.value)}
      className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono
                 focus:outline-none focus:border-indigo-500"
    />
  </div>
))}
```

### TODO 7 — Update `updateNodeParam` to support nested `_output_aliases` key
```ts
// TODO: frontend/nested-param-update
if (key.startsWith("_output_aliases.")) {
  const portId = key.split(".")[1];
  updateNode(nodeId, (data) => ({
    ...data,
    paramValues: {
      ...data.paramValues,
      _output_aliases: {
        ...(data.paramValues._output_aliases ?? {}),
        [portId]: value,
      }
    }
  }));
}
```

### TODO 8 — Validate alias as a valid Python identifier
Show an inline red border + tooltip if the alias isn't a valid Python identifier:
```ts
// TODO: frontend/alias-validation
const PYTHON_IDENTIFIER = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const isValid = !alias || (PYTHON_IDENTIFIER.test(alias) && !PYTHON_KEYWORDS.has(alias));
```

---

## 🐛 Known Bugs

- Check shapes cant be disabled once enabled
- Separate testing modules and scripts are need to check these features for every block and all situations.