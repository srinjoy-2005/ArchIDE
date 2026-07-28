from models import CompileRequest, Node, Edge, PortDef
from typing import List, Dict, Tuple, Any, Optional
from blocks import get_block_by_id
import re

# ─── Helpers ────────────────────────────────────────────────────────────────

def _sanitize(label: str) -> str:
    """Turn a block label into a valid Python identifier fragment."""
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # replace non-alphanumeric runs with _
    s = s.strip("_")
    return s or "var"

def _resolve_block_id(node: Node) -> str:
    block_id = getattr(node.data, "block_id", "").lower()
    if not block_id:
        label = node.data.label
        block_id = "split" if "Split" in label else label.split()[0].lower()
    return block_id


# ─── Topological Sort ────────────────────────────────────────────────────────

def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
    adj = {node.id: [] for node in nodes}
    in_degree = {node.id: 0 for node in nodes}
    node_map = {node.id: node for node in nodes}

    for edge in edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue = [node_id for node_id in in_degree if in_degree[node_id] == 0]
    sorted_nodes = []

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


# ─── Shape Inference Pass ─────────────────────────────────────────────────────

class ShapeError(Exception):
    """Raised when a shape mismatch is detected during static analysis."""
    def __init__(self, message: str, node_id: str, node_label: str):
        super().__init__(message)
        self.node_id = node_id
        self.node_label = node_label


def shape_inference_pass(
    sorted_nodes: List[Node], edges: List[Edge]
) -> Dict[str, Tuple]:
    """
    Traverse the graph in topological order, propagate concrete tensor shapes,
    and raise ShapeError immediately if a mismatch is detected.

    A dimension value of the string "ANY" means the dimension is dynamically
    determined at runtime. Nodes that encounter an ANY upstream dimension
    skip strict validation and propagate ANY further downstream.
    """
    # Maps "{node_id}_{port_id}" -> shape tuple, e.g. (1, 64, 112, 112)
    tensor_shapes: Dict[str, Tuple] = {}

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        block = get_block_by_id(block_id)
        if not block:
            continue

        # Gather incoming shapes from already-resolved upstream ports
        incoming_shapes: Dict[str, Tuple] = {}
        for port in block.definition.inputs:
            incoming_edge = next(
                (e for e in edges if e.target == node.id and e.targetHandle == port.id),
                None,
            )
            if incoming_edge:
                source_key = f"{incoming_edge.source}_{incoming_edge.sourceHandle}"
                incoming_shapes[port.id] = tensor_shapes.get(source_key, ("ANY",))
            else:
                incoming_shapes[port.id] = ("ANY",)

        # Delegate shape inference to the block class and catch any mismatch
        try:
            out_shapes = block.infer_shapes(incoming_shapes, node.data.paramValues)
        except ValueError as exc:
            raise ShapeError(
                message=str(exc),
                node_id=node.id,
                node_label=node.data.label,
            ) from exc

        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape

    return tensor_shapes


# ─── Code Generation ──────────────────────────────────────────────────────────

def _build_input_var_map(
    sorted_nodes: List[Node],
) -> Dict[str, str]:
    """
    Build a disambiguated map of input-node output keys -> Python variable names.

    Two input nodes that share the same label (e.g. both called "Input") get
    unique names: x_input, x_input_2, x_input_3, …
    """
    INPUT_IDS = {"input", "gourav"}

    # Collect all input nodes in topological order
    input_nodes = [
        n for n in sorted_nodes if _resolve_block_id(n) in INPUT_IDS
    ]

    label_counts: Dict[str, int] = {}
    var_map: Dict[str, str] = {}  # "{node_id}_out" -> variable name

    for node in input_nodes:
        base = _sanitize(node.data.label)  # e.g. "input", "image", "mask"
        count = label_counts.get(base, 0) + 1
        label_counts[base] = count

        if count == 1:
            var_name = f"x_{base}"
        else:
            var_name = f"x_{base}_{count}"

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
      2. port.var_hint (block-defined semantic name, e.g. "sum", "conv_feat")
      3. Fallback: x_{short_node_id}

    Disambiguates duplicates by appending a counter:
      conv_feat, conv_feat_2, conv_feat_3, …
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

    # Build forward() signature from the same unique variable map
    INPUT_IDS = {"input", "gourav"}
    forward_arg_vars = [
        var_map[f"{n.id}_out"]
        for n in sorted_nodes
        if _resolve_block_id(n) in INPUT_IDS
    ]
    forward_lines = [f"    def forward(self, {', '.join(forward_arg_vars)}):"]

    # ── Step 4: Edge lookup table ─────────────────────────────────────────────
    # target_node_id + target_handle -> source_node_id + source_handle
    input_to_source: Dict[str, str] = {}
    for edge in edges:
        target_key = f"{edge.target}_{edge.targetHandle}"
        source_key = f"{edge.source}_{edge.sourceHandle}"
        input_to_source[target_key] = source_key

    # ── Step 5: Walk the sorted graph and emit code ───────────────────────────
    # hint_counts tracks how many times each var_hint base has been used,
    # enabling automatic disambiguation: sum, sum_2, sum_3, …
    hint_counts: Dict[str, int] = {}
    for node in sorted_nodes:
        node_id = node.id
        block_id = _resolve_block_id(node)
        block = get_block_by_id(block_id)
        if not block:
            continue

        params = node.data.paramValues

        # Skip input nodes – their variable names were already seeded into var_map
        if block_id in INPUT_IDS:
            continue

        # Resolve input variable names from var_map
        input_vars: Dict[str, str] = {}
        for port in block.definition.inputs:
            src_key = input_to_source.get(f"{node_id}_{port.id}")
            input_vars[port.id] = var_map.get(src_key, "None") if src_key else "None"

        # Build output variable names using var_hint / user aliases
        output_vars: Dict[str, str] = {}
        for port in block.definition.outputs:
            out_var = _build_output_var(port, node_id, params, hint_counts)
            output_vars[port.id] = out_var
            var_map[f"{node_id}_{port.id}"] = out_var

        # Emit __init__ line (stateful layers only)
        if not block.definition.is_functional:
            init_code = block.emit_init(node_id, params)
            if init_code:
                init_lines.append(f"        {init_code}")

        # Emit forward() line
        forward_code = block.emit_forward(node_id, input_vars, output_vars, params)
        if forward_code:
            forward_lines.append(f"        {forward_code}")

    # ── Step 6: Handle empty bodies ───────────────────────────────────────────
    if not init_lines:
        init_lines.append("        pass")
    if len(forward_lines) == 1:
        forward_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)

    return "\n".join(code)
