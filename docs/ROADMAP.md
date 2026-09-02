# ArchiDE Master Roadmap & Project Checklist

A unified, actionable tracking document for the ArchiDE visual deep learning architecture builder.

---

## 📊 Feature Status Overview

| Milestone | Status | Description |
|---|:---:|---|
| **Core Architecture & OOP IR** | `COMPLETED` | Polymorphic `BaseBlock` architecture with 28 blocks across 8 categories. |
| **Compiler & Shape Inference** | `COMPLETED` | Kahn's topo-sort, 3-stage compile pipeline, and `/api/check` 422 static shape validation. |
| **Visual Canvas & Properties** | `COMPLETED` | React Flow uncontrolled mode, 3-section inspector, custom variable names, and connection rules. |
| **Variadic Ports & Multi-Outputs**| `COMPLETED` | Variadic ports (`Add`, `Mul`, `Concat`, `Output`) and orphan edge resilience. |
| **Custom Modules & Multi-Graph Compilation** | `DESIGNED` | Full spec in [`custom_modules_architecture.md`](backend/custom_modules_architecture.md); dual-target codegen, module registry, and cycle detection. |
| **Automated Testing Suite** | `IN PROGRESS` | 16 backend unit/compiler tests active; frontend and full 28-block coverage pending. |
| **Save/Load & Model Export** | `PLANNED` | JSON graph serialization, project save/load, and direct `.py`/ONNX export. |

---

## 1. 🏗️ Core Engine & Compiler Pipeline

- [x] **Monolithic to OOP IR Migration**: Refactored `compiler.py` into polymorphic classes in `backend/blocks/` inheriting from `BaseBlock`.
- [x] **Pydantic Data Contracts**: Canonical `BlockDef`, `PortDef`, `ParamDef`, and `CompileRequest` schemas in `backend/models.py`.
- [x] **Kahn's Topological Sort**: Deterministic sorting with cycle detection (HTTP 400).
- [x] **Static Shape Inference Pass**: Concrete shape calculation with `-1` auto-inference and `ShapeError` (HTTP 422).
- [x] **Variable Disambiguation & Semantic Hints**: Pre-seeded input variables (`x_input`, `x_input_2`), user variable overrides, and `var_hint` naming (`conv_feat`, `sum`, `probs`).
- [x] **Orphan Edge Resilience**: Compiler filters out edges referencing deleted nodes to prevent `KeyError` crashes.
- [x] **Variadic Operation Support**: Single variadic port (`is_list=True`) on `Add`, `Multiply`, `Concat`, and `Output`.
- [x] **Multi-Output Aggregation**: Unified forward return tuple (`return out_1, out_2`) across single or multiple `Output` blocks.
- [ ] **Dynamic AST Validation**: Direct PyTorch execution verification of generated code using dummy tensor forward passes.

---

## 2. 🎨 Visual Canvas & Frontend UI

- [x] **Uncontrolled React Flow Architecture**: Strict `defaultNodes`/`defaultEdges` model in `src/app/page.tsx`.
- [x] **Custom Node Components**: `CustomNode.tsx` with dynamic category accent colors, handle badges, and hover tooltips.
- [x] **Midpoint Tensor Edges**: `TensorEdge.tsx` with shape dimension chips and bezier curves.
- [x] **Three-Section Properties Panel**: Tensor Shapes (read-only), Hyperparameters (editable), and Collapsible Advanced Properties.
- [x] **Connection Validation (`isValidConnection`)**:
  - Strict 1-to-1 incoming connections for single-input ports (`Linear`, `Conv2D`, `ReLU`, `Sub.in_a`, etc.).
  - Multi-incoming connections for variadic ports (`Add`, `Concat`, `Multiply`, `Output`).
- [x] **Inspector Connected Inputs List**: Displays real-time upstream tensor names and counts for selected blocks.
- [x] **Shape Error Visualizer**: Real-time red highlight and warning badges on offending nodes upon shape mismatch.
- [x] **Keyboard Deletion Cleanup**: `onNodesDelete` and `handleDelete` automatically prune connected edges.
- [ ] **Canvas Shape Check Toggle**: Ability to toggle static shape overlays and error badges on/off directly from the canvas header.
- [ ] **Parameter Input Validation**: Enhanced frontend validators (e.g. integer range bounds, positive kernel sizes).

---

## 3. 🧪 Comprehensive Testing Suite & Robustness Checklist

### A. Backend Block Unit Tests (`backend/tests/test_blocks.py`)
- [x] `LinearBlock`: Standard shape propagation, `-1` auto-inference of `in_features`, shape mismatch ValueError.
- [x] `Conv2DBlock`: Dilated spatial formulas, `-1` auto-inference of `in_channels`, negative spatial dimensions catch.
- [x] `InputBlock`: Tuple string parsing and fallback handling.
- [ ] **All 28 Block Coverage**:
  - [ ] Activations (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `SiLU`, `LeakyReLU`, `ELU`).
  - [ ] Pooling (`MaxPool2D`, `AvgPool2D`, `AdaptiveAvgPool2D`).
  - [ ] Normalization (`BatchNorm2D`, `LayerNorm`, `GroupNorm`, `RMSNorm`).
  - [ ] Tensor Ops (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `MatMul`, `Concat`, `Split`, `Reshape`, `Flatten`, `Transpose`, `Squeeze`, `Unsqueeze`, `Dropout`).
  - [ ] Multi-Head Attention (`MultiheadAttentionBlock`).
- [ ] **Auto-Inference Boundary Tests**: Assert all blocks with auto-inferred parameters correctly populate `paramValues` without manual entry.

### B. Compiler & Graph Robustness Tests (`backend/tests/test_compiler.py`)
- [x] Topo-sort on linear graph.
- [x] Topo-sort cycle detection on circular graph.
- [x] Orphan edge tolerance (edges pointing to deleted node IDs).
- [x] Variadic `Add` with 3+ input branches.
- [x] Multi-tensor `Output` return aggregation.
- [x] PyTorch code syntax compilation (`compile(code, "<string>", "exec")`).
- [ ] **Complex Branching Topologies**:
  - [ ] Residual skip connections (ResNet bottleneck block with Add).
  - [ ] Multi-branch merge via `Concat` along dim 1.
  - [ ] Multi-head outputs (e.g. classification logits + bounding box regression).
  - [ ] Disconnected subgraph handling (warning or isolated block omission).

### C. API Integration Tests (`backend/tests/test_api.py`)
- [x] `GET /api/blocks` schema validation.
- [x] `POST /api/check` returning HTTP 422 with `ShapeError` detail on non-broadcastable tensor shapes.
- [x] `POST /api/compile` returning HTTP 400 on cyclic graph.
- [x] `POST /api/compile` returning HTTP 200 with valid PyTorch model code.
- [ ] `GET /api/blocks/{id}/docs` returning Markdown docstrings for all registered blocks.
- [ ] Malformed payload handling (missing node data, invalid JSON types).

### D. Frontend Automated Testing
- [ ] **Component Tests (Jest / React Testing Library)**:
  - [ ] `CustomNode`: Renders correct handle count, variable names, and error badge states.
  - [ ] `PropertiesPanel`: Correctly mutates node parameter state and dynamic input lists.
  - [ ] `TensorEdge`: Formats shape chip labels accurately.
- [ ] **E2E Canvas Tests (Playwright)**:
  - [ ] Drag-and-drop from sidebar palette to canvas.
  - [ ] Drawing edges between valid vs. invalid ports.
  - [ ] Deleting nodes with Backspace and verifying edge cleanup.

---

## 4. 📦 Custom Submodules & Multi-Graph Compilation

> [!NOTE]
> Detailed technical design and pipeline specification: [`docs/backend/custom_modules_architecture.md`](backend/custom_modules_architecture.md)

- [x] **Compiler-Linker Architectural Specification**: Detailed analysis of monolithic single-file vs. modular Python package generation with concrete dual-target codegen pipeline.
- [ ] **Module Registry & Interface Extraction**: Automatic extraction of port signatures and parameter contracts from custom module tabs/files.
- [ ] **Project DAG & Cycle Detection**: Tarjan/Kahn's inter-module dependency resolution with formatted cyclic path error reporting.
- [ ] **Two-Tier Shape Inference**: Inter-module shape propagation across custom module boundaries via interface contracts.
- [ ] **Dual-Target Codegen Pipeline**:
  - [ ] Target A (Default): Modular Python package emitter with clean relative imports and `__init__.py`.
  - [ ] Target B (Export): Monolithic linker inlining all submodule classes in topological order into a single `model.py`.
- [ ] **Visual IDE Integration**:
  - [ ] Dynamic "My Modules" palette population from project tabs/XML files.
  - [ ] Export modal offering "Export as Single File" and "Export as Python Package".
  - [ ] Source mapping (`.py.map`) linking generated code errors to visual node IDs.
- [ ] **Incremental Build Engine**: Interface vs. body hashing (`.arch_cache/`) for sub-second re-compilation on canvas edits.

---

## 5. 🚀 Future Roadmap & Scalability

- [ ] **Project Persistence (Save & Load)**:
  - Serialize full graph state (nodes, edges, viewport, parameter overrides) to `.archide.json`.
  - File picker upload and local browser storage autosave.
- [ ] **Export Options**:
  - Direct `.py` file download button in the code panel.
  - ONNX / TorchScript graph export generator.
- [ ] **Block Multiplicity & Subgraphs**:
  - Draggable `Repeat` wrapper or `repeat_count` parameter to stack $N$ identical blocks (e.g. 12x Transformer layers).
  - Collapsible Subgraph / Macro node grouping.
- [ ] **Expanded Block Registry**:
  - Transformer blocks (Self-Attention, Cross-Attention, MLP FeedForward).
  - Recurrent blocks (LSTM, GRU).
  - Loss functions (`CrossEntropyLoss`, `MSELoss`, `CosineSimilarityLoss`).
