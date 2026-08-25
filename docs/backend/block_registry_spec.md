# ArchIDE — Block Registry Specification

This document provides the formal specification for how draggable neural network blocks, ports, parameters, and code emission protocols are defined in ArchIDE.

> [!NOTE]
> **Source Files**:
> - Pydantic Schemas: [`backend/models.py`](../../backend/models.py#L30-L57)
> - Block Base Class & Interface: [`backend/blocks/base.py`](../../backend/blocks/base.py#L10-L38)
> - Registry Aggregation & Lookup: [`backend/blocks/__init__.py`](../../backend/blocks/__init__.py#L16-L64) and [`backend/registry.py`](../../backend/registry.py)

---

## 1. Core Data Models (`backend/models.py`)

Every block is declared via a `BlockDef` Pydantic model returned by the block's `definition` property:

- **`PortDef`** ([`backend/models.py`](../../backend/models.py#L30-L39)):
  - `id: str`: Unique handle identifier on the node (e.g., `"in"`, `"out"`, `"in_a"`, `"out_1"`).
  - `name: str`: UI label shown next to the port handle.
  - `type: str`: Port data type (default: `"tensor"`).
  - `is_list: bool`: Whether the port accepts multiple incoming connections (e.g. variadic inputs).
  - `var_hint: Optional[str]`: Suggested variable name prefix during PyTorch code generation (e.g. `"fc_out"`, `"conv_feat"`, `"probs"`).

- **`ParamDef`** ([`backend/models.py`](../../backend/models.py#L40-L47)):
  - `name: str`: Parameter name passed into `paramValues` dictionary.
  - `type: str`: Type identifier (`"int"`, `"float"`, `"string"`, `"bool"`).
  - `default: Any`: Default initial value.
  - `read_only: bool`: When `True`, shown greyed-out / non-editable in the UI.
  - `section: str`: Property grouping (`"basic"` | `"advanced"` | `"shape"`).
  - `description: str`: Tooltip description rendered in the inspector.

- **`BlockDef`** ([`backend/models.py`](../../backend/models.py#L48-L57)):
  - `id: str`: Unique identifier (e.g. `"linear"`, `"conv2d"`, `"relu"`).
  - `name: str`: Display name (e.g. `"Linear"`, `"Conv2D"`).
  - `category: str`: Category for sidebar grouping (`"Core Layers"`, `"Activations"`, `"Pooling"`, `"Normalization"`, `"Shape Ops"`, `"Tensor Ops"`, `"Trig"`, `"Generators"`).
  - `color: str`: Accent hex color for canvas nodes.
  - `is_functional: bool`: `True` for stateless/functional operations (inlined into `forward()` only); `False` for stateful layers requiring `nn.Module` instantiation in `__init__()`.
  - `inputs: List[PortDef]`: Target handles rendered on node left side.
  - `outputs: List[PortDef]`: Source handles rendered on node right side.
  - `params: List[ParamDef]`: Configurable parameters shown in Properties Panel.

---

## 2. Block Implementation Protocol (`BaseBlock`)

Every block inherits from `BaseBlock` ([`backend/blocks/base.py`](../../backend/blocks/base.py#L10-L38)) and implements:

1. **`definition -> BlockDef`**: Returns the Pydantic schema for the block.
2. **`infer_shapes(input_shapes: Dict[str, Tuple], params: Dict[str, Any]) -> Dict[str, Tuple]`**: Computes output port shapes from input port shapes. May mutate `params` for auto-inference (e.g. `in_features=-1`). Raises `ValueError` on shape mismatch.
3. **`emit_init(node_id: str, params: Dict[str, Any]) -> str`**: Emits `self.layer_{node_id} = nn.<Module>(...)` for stateful layers, or `""` for functional ops.
4. **`emit_forward(node_id: str, input_vars: Dict[str, str], output_vars: Dict[str, str], params: Dict[str, Any]) -> str`**: Emits Python forward pass computation string.
5. **`docs() -> Dict[str, str]`**: Returns `{"intro": "...", "details": "..."}` markdown documentation.

---

## 3. Frontend & Backend Interaction

- **Palette Fetch**: On initialization, the frontend calls `GET /api/blocks` ([`backend/main.py`](../../backend/main.py#L28-L30)), which serializes all block definitions from `get_all_block_defs()` ([`backend/blocks/__init__.py`](../../backend/blocks/__init__.py#L61-L63)).
- **Block Docs Tooltips**: The frontend requests markdown documentation on demand via `GET /api/blocks/{block_id}/docs` ([`backend/main.py`](../../backend/main.py#L32-L38)).
- **Compilation & Validation**: Graph payloads (`nodes` and `edges`) are sent via `POST /api/check` for static shape analysis or `POST /api/compile` for PyTorch code generation ([`backend/main.py`](../../backend/main.py#L40-L90)).
