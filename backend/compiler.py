import re
from models import Node, Edge
from typing import List, Dict, Tuple, Any, Optional
from blocks import get_block_by_id


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
# Helper: sanitize a user label into a valid Python identifier
# ---------------------------------------------------------------------------

def _label_to_identifier(label: str) -> str:
    """Turn a user-provided node name into a clean Python snake_case identifier."""
    s = label.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    if not s or s[0].isdigit():
        s = "x_" + s
    return s


# ---------------------------------------------------------------------------
# Shape Inference Pass
# ---------------------------------------------------------------------------

def shape_inference_pass(
    sorted_nodes: List[Node],
    edges: List[Edge],
) -> Dict[str, Dict[str, Tuple]]:
    """
    Returns a dict mapping  node_id -> {port_id -> shape_tuple}.
    Raises ValueError on the first shape mismatch found.
    """
    # Maps f"{node_id}_{port_id}" -> shape tuple
    tensor_shapes: Dict[str, Tuple] = {}
    # Per-node output shapes for rich reporting
    node_out_shapes: Dict[str, Dict[str, Tuple]] = {}

    for node in sorted_nodes:
        block_id = node.data.block_id.strip().lower()
        block = get_block_by_id(block_id)
        if not block:
            continue

        # Gather incoming shapes from the propagated tensor_shapes map
        incoming: Dict[str, Tuple] = {}
        for port in block.definition.inputs:
            edge = next(
                (e for e in edges if e.target == node.id and e.targetHandle == port.id),
                None
            )
            if edge:
                src_key = f"{edge.source}_{edge.sourceHandle}"
                incoming[port.id] = tensor_shapes.get(src_key, ("ANY",))
            else:
                incoming[port.id] = ("ANY",)

        # Block performs its own shape validation and returns output shapes
        out_shapes = block.infer_shapes(incoming, node.data.paramValues)

        # Store for downstream propagation
        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape

        node_out_shapes[node.id] = out_shapes

    return node_out_shapes


# ---------------------------------------------------------------------------
# Code Generation Pass
# ---------------------------------------------------------------------------

def generate_pytorch_code(sorted_nodes: List[Node], edges: List[Edge]) -> str:
    # -----------------------------------------------------------------------
    # 1. Build edge lookup: f"{target_node_id}_{targetHandle}" -> source_key
    # -----------------------------------------------------------------------
    input_to_source: Dict[str, str] = {}
    for edge in edges:
        target_key = f"{edge.target}_{edge.targetHandle}"
        source_key = f"{edge.source}_{edge.sourceHandle}"
        input_to_source[target_key] = source_key

    # -----------------------------------------------------------------------
    # 2. Variable name map: source_key -> python variable name
    # -----------------------------------------------------------------------
    var_map: Dict[str, str] = {}

    # -----------------------------------------------------------------------
    # 3. Build forward() signature from Input nodes
    #    Use deduplicated, sanitized labels. Sequential counter as tiebreaker.
    # -----------------------------------------------------------------------
    input_nodes = [n for n in sorted_nodes if n.data.block_id == "input"]
    
    seen_bases: Dict[str, int] = {}   # base_name -> count of occurrences
    forward_arg_names: List[str] = []

    for node in input_nodes:
        base = _label_to_identifier(node.data.label)
        if base in seen_bases:
            seen_bases[base] += 1
            arg_name = f"{base}_{seen_bases[base]}"
        else:
            seen_bases[base] = 0
            arg_name = base
        forward_arg_names.append(arg_name)
        # Register this input's output port in the var_map
        var_map[f"{node.id}_out"] = arg_name

    # -----------------------------------------------------------------------
    # 4. Assign sequential clean variable names (x_0, x_1, …) to all
    #    non-input, non-output nodes using the Kahn order guarantee.
    # -----------------------------------------------------------------------
    seq_counter = 0
    for node in sorted_nodes:
        block_id = node.data.block_id.strip().lower()
        if block_id in ("input", "output"):
            continue
        block = get_block_by_id(block_id)
        if not block:
            continue

        # For each output port, assign a sequential name
        for port in block.definition.outputs:
            if len(block.definition.outputs) == 1:
                var_name = f"x_{seq_counter}"
            else:
                var_name = f"x_{seq_counter}_{port.id}"
            var_map[f"{node.id}_{port.id}"] = var_name

        seq_counter += 1

    # -----------------------------------------------------------------------
    # 5. Collect Output nodes' return values (for single return statement)
    # -----------------------------------------------------------------------
    output_nodes = [n for n in sorted_nodes if n.data.block_id == "output"]
    return_vars: List[str] = []
    for node in output_nodes:
        src_key = input_to_source.get(f"{node.id}_in")
        ret_var = var_map.get(src_key, "None") if src_key else "None"
        return_vars.append(ret_var)

    # -----------------------------------------------------------------------
    # 6. Code emission
    # -----------------------------------------------------------------------
    code        = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    init_lines:    List[str] = []
    forward_lines: List[str] = [f"    def forward(self, {', '.join(forward_arg_names)}):"]

    for node in sorted_nodes:
        block_id = node.data.block_id.strip().lower()
        if block_id in ("input", "output"):
            continue  # Input vars already registered; Output return handled separately

        block = get_block_by_id(block_id)
        if not block:
            forward_lines.append(f"        # WARNING: unknown block '{block_id}' — skipped")
            continue

        params = node.data.paramValues

        # Resolve input variable names from the var_map
        input_vars: Dict[str, str] = {}
        for port in block.definition.inputs:
            src_key = input_to_source.get(f"{node.id}_{port.id}")
            input_vars[port.id] = var_map.get(src_key, "None") if src_key else "None"

        # Resolve output variable names
        output_vars: Dict[str, str] = {
            port.id: var_map[f"{node.id}_{port.id}"]
            for port in block.definition.outputs
            if f"{node.id}_{port.id}" in var_map
        }

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
    # 7. Emit single return statement
    # -----------------------------------------------------------------------
    if return_vars:
        forward_lines.append(f"        return {', '.join(return_vars)}")

    # Fallback stubs for empty bodies
    if not init_lines:
        init_lines.append("        pass")
    if len(forward_lines) == 1:
        forward_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    return "\n".join(code)
