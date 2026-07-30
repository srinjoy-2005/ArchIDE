from models import CompileRequest, Node, Edge, PortDef
from typing import List, Dict, Tuple, Any, Optional
from blocks import get_block_by_id
import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(label: str) -> str:
    """Turn a block label into a valid Python identifier fragment."""
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "var"

def _resolve_block_id(node: Node) -> str:
    """Get the canonical block_id from a node, falling back gracefully."""
    block_id = getattr(node.data, "block_id", "").lower().strip()
    return block_id or "unknown"

def _label_to_identifier(label: str) -> str:
    """Turn a user-provided node name into a clean Python snake_case identifier."""
    s = _sanitize(label)
    if not s or s[0].isdigit():
        s = "x_" + s
    return s


# ---------------------------------------------------------------------------
# Shape Error
# ---------------------------------------------------------------------------

class ShapeError(Exception):
    """Raised when a shape mismatch is detected during static analysis."""
    def __init__(self, message: str, node_id: str, node_label: str):
        super().__init__(message)
        self.node_id = node_id
        self.node_label = node_label


# ---------------------------------------------------------------------------
# Topological Sort (Kahn's Algorithm)
# ---------------------------------------------------------------------------

def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
    adj       = {node.id: [] for node in nodes}
    in_degree = {node.id: 0  for node in nodes}
    node_map  = {node.id: node for node in nodes}

    for edge in edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_nodes: List[Node] = []

    while queue:
        curr_id = queue.pop(0)
        sorted_nodes.append(node_map[curr_id])
        for neighbor in adj[curr_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Cycle detected in graph! Cannot compile.")

    return sorted_nodes


# ---------------------------------------------------------------------------
# Shape Inference Pass
# ---------------------------------------------------------------------------

def shape_inference_pass(
    sorted_nodes: List[Node],
    edges: List[Edge],
) -> Tuple[Dict[str, Dict[str, Tuple]], Dict[str, Dict[str, Any]]]:
    """
    Traverse the graph in topological order, propagate concrete tensor shapes,
    and raise ShapeError immediately on the first mismatch detected.

    Returns a dict: node_id -> {port_id -> shape_tuple}
    A dimension value of the string "ANY" means the dimension is dynamically
    determined at runtime.
    """
    # Maps f"{node_id}_{port_id}" -> shape tuple, e.g. (1, 64, 112, 112)
    tensor_shapes: Dict[str, Tuple] = {}
    # Per-node output shapes for rich reporting
    node_out_shapes: Dict[str, Dict[str, Tuple]] = {}
    # Track any parameters the blocks auto-inferred and mutated
    updated_node_params: Dict[str, Dict[str, Any]] = {}

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        block = get_block_by_id(block_id)
        if not block:
            continue

        # Gather incoming shapes from already-resolved upstream ports
        incoming: Dict[str, Tuple] = {}
        for port in block.definition.inputs:
            incoming_edge = next(
                (e for e in edges if e.target == node.id and e.targetHandle == port.id),
                None,
            )
            if incoming_edge:
                src_key = f"{incoming_edge.source}_{incoming_edge.sourceHandle}"
                incoming[port.id] = tensor_shapes.get(src_key, ("ANY",))
            else:
                incoming[port.id] = ("ANY",)

        # Keep track of original params to detect auto-inference mutations
        original_params = dict(node.data.paramValues)

        # Delegate shape inference to the block class and catch any mismatch
        try:
            out_shapes = block.infer_shapes(incoming, node.data.paramValues)
        except ValueError as exc:
            raise ShapeError(
                message=str(exc),
                node_id=node.id,
                node_label=node.data.label,
            ) from exc

        # Store for downstream propagation
        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape

        # Check if the block mutated any parameters (e.g. auto-inferred in_features)
        for k, v in node.data.paramValues.items():
            if k not in original_params or original_params[k] != v:
                if node.id not in updated_node_params:
                    updated_node_params[node.id] = {}
                updated_node_params[node.id][k] = v

        # Combine incoming and out_shapes so both are returned to the frontend for display
        combined_shapes = {**incoming, **out_shapes}
        node_out_shapes[node.id] = combined_shapes

    return node_out_shapes, updated_node_params


# ---------------------------------------------------------------------------
# Code Generation Pass
# ---------------------------------------------------------------------------

def _build_input_var_map(sorted_nodes: List[Node]) -> Dict[str, str]:
    """
    Build a disambiguated map of input-node output keys -> Python variable names.
    Two input nodes that share the same label (e.g. both called "Input") get
    unique names: x_input, x_input_2, x_input_3, ...
    """
    INPUT_IDS = {"input"}
    input_nodes = [n for n in sorted_nodes if _resolve_block_id(n) in INPUT_IDS]

    label_counts: Dict[str, int] = {}
    var_map: Dict[str, str] = {}  # "{node_id}_out" -> variable name

    for node in input_nodes:
        base = _label_to_identifier(node.data.label)
        count = label_counts.get(base, 0) + 1
        label_counts[base] = count

        if count == 1:
            var_name = base if base.startswith("x_") else f"x_{base}"
        else:
            base_clean = base[2:] if base.startswith("x_") else base
            var_name = f"x_{base_clean}_{count}"

        var_map[f"{node.id}_out"] = var_name

    return var_map


def _build_output_var(
    port: PortDef,
    node_id: str,
    params: Dict[str, Any],
    hint_counts: Dict[str, int],
) -> str:
    """
    Choose a unique, readable variable name for an output port.
    Priority order:
      1. User alias from paramValues["_output_aliases"][port.id]
      2. port.var_hint
      3. Fallback: x_{short_node_id}
    """
    aliases = params.get("_output_aliases", {})
    user_alias = aliases.get(port.id, "") if isinstance(aliases, dict) else ""

    if user_alias:
        base = _sanitize(user_alias)
    elif port.var_hint:
        base = port.var_hint
    else:
        base = f"x_{node_id.replace('-', '_')[:8]}"

    count = hint_counts.get(base, 0) + 1
    hint_counts[base] = count
    return base if count == 1 else f"{base}_{count}"


def generate_pytorch_code(sorted_nodes: List[Node], edges: List[Edge]) -> str:
    # ── Step 1: Static shape check (raises ShapeError on mismatch) ──────────
    shape_inference_pass(sorted_nodes, edges)

    # ── Step 2: Boilerplate ──────────────────────────────────────────────────
    code = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    init_lines: List[str] = []

    # ── Step 3: Build unique variable map for all input nodes ────────────────
    var_map = _build_input_var_map(sorted_nodes)

    INPUT_IDS = {"input"}
    forward_arg_names = [
        var_map[f"{n.id}_out"]
        for n in sorted_nodes
        if _resolve_block_id(n) in INPUT_IDS
    ]
    forward_lines: List[str] = [f"    def forward(self, {', '.join(forward_arg_names)}):"]

    # ── Step 4: Edge lookup table ─────────────────────────────────────────────
    input_to_source: Dict[str, str] = {}
    for edge in edges:
        target_key = f"{edge.target}_{edge.targetHandle}"
        source_key = f"{edge.source}_{edge.sourceHandle}"
        input_to_source[target_key] = source_key

    # ── Step 5: Walk the sorted graph and emit code ───────────────────────────
    hint_counts: Dict[str, int] = {}
    return_vars: List[str] = []

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        
        # Skip input nodes as they are already handled in the signature
        if block_id in INPUT_IDS:
            continue

        block = get_block_by_id(block_id)
        if not block:
            forward_lines.append(f"        # WARNING: unknown block '{block_id}' — skipped")
            continue

        params = node.data.paramValues

        # Resolve input variable names
        input_vars: Dict[str, str] = {}
        for port in block.definition.inputs:
            src_key = input_to_source.get(f"{node.id}_{port.id}")
            input_vars[port.id] = var_map.get(src_key, "None") if src_key else "None"

        # Handle Output nodes explicitly
        if block_id == "output":
            in_var = input_vars.get("in", "None")
            if in_var != "None":
                return_vars.append(in_var)
            continue

        # Build output variable names for this node
        output_vars: Dict[str, str] = {}
        for port in block.definition.outputs:
            out_var = _build_output_var(port, node.id, params, hint_counts)
            output_vars[port.id] = out_var
            var_map[f"{node.id}_{port.id}"] = out_var

        # Emit __init__ line (stateful layers only)
        if not block.definition.is_functional:
            init_code = block.emit_init(node.id, params)
            if init_code:
                init_lines.append(f"        {init_code}")

        # Emit forward line
        forward_code = block.emit_forward(node.id, input_vars, output_vars, params)
        if forward_code:
            forward_lines.append(f"        {forward_code}")

    # -----------------------------------------------------------------------
    # 8. Emit single return statement
    # -----------------------------------------------------------------------
    if return_vars:
        forward_lines.append(f"        return {', '.join(return_vars)}")
    else:
        # If there are no explicit output blocks, we could just pass or return None.
        if len(forward_lines) == 1:
            forward_lines.append("        pass")

    # Fallback stubs for empty bodies
    if not init_lines:
        init_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    return "\n".join(code)
