from models import CompileRequest, Node, Edge, PortDef, BlockDef
from typing import List, Dict, Tuple, Any, Optional
from blocks import get_block_by_id, BaseBlock
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
    def __init__(self, message: str, node_id: str, node_label: str, edge_ids: List[str] = None):
        super().__init__(message)
        self.node_id = node_id
        self.node_label = node_label
        self.edge_ids = edge_ids or []


# ---------------------------------------------------------------------------
# Topological Sort (Kahn's Algorithm)
# ---------------------------------------------------------------------------

def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
    node_ids = {node.id for node in nodes}
    valid_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    adj       = {node.id: [] for node in nodes}
    in_degree = {node.id: 0  for node in nodes}
    node_map  = {node.id: node for node in nodes}

    for edge in valid_edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_nodes: List[Node] = []

    while queue:
        queue.sort() # Ensure deterministic resolution
        curr_id = queue.pop(0)
        sorted_nodes.append(node_map[curr_id])
        for neighbor in adj[curr_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Cycle detected in graph! Cannot compile.")

    return sorted_nodes


def topological_sort_graphs(graphs: Dict[str, Any]) -> List[str]:
    adj = {gid: [] for gid in graphs}
    in_degree = {gid: 0 for gid in graphs}
    for gid, data in graphs.items():
        for node in data.nodes:
            block_id = _resolve_block_id(node)
            if block_id == "custom_module":
                dep_id = getattr(node.data, "custom_module_id", "")
                if dep_id in graphs:
                    adj[dep_id].append(gid)
                    in_degree[gid] += 1
    
    queue = [gid for gid, deg in in_degree.items() if deg == 0]
    sorted_gids = []
    while queue:
        queue.sort()
        curr = queue.pop(0)
        sorted_gids.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(sorted_gids) != len(graphs):
        raise ValueError("Cycle detected in custom module dependencies!")
    return sorted_gids


# ---------------------------------------------------------------------------
# Custom Module Support
# ---------------------------------------------------------------------------

def _get_custom_block_def(gid: str, graphs: Dict[str, Any]) -> BlockDef:
    graph_data = graphs[gid]
    input_nodes = [n for n in graph_data.nodes if getattr(n.data, "block_id", "") == "input"]
    output_nodes = [n for n in graph_data.nodes if getattr(n.data, "block_id", "") == "output"]
    
    inputs = [PortDef(id=n.id, name=n.data.label) for n in input_nodes]
    outputs = [PortDef(id=n.id, name=n.data.label) for n in output_nodes]
    
    return BlockDef(
        id=f"custom_module_{gid}",
        name=graph_data.name,
        category="Custom Modules",
        color="#000",
        is_functional=False,
        inputs=inputs,
        outputs=outputs,
        params=[]
    )

class CustomModuleBlock(BaseBlock):
    def __init__(self, definition: BlockDef, class_name: str):
        self._definition = definition
        self.class_name = class_name
        
    @property
    def definition(self) -> BlockDef:
        return self._definition
        
    def emit_init(self, node_id: str, params: dict) -> str:
        var_name = f"self.custom_{node_id.replace('-', '_')[:8]}"
        return f"{var_name} = {self.class_name}()"
        
    def emit_forward(self, node_id: str, input_vars: dict, output_vars: dict, params: dict) -> str:
        var_name = f"self.custom_{node_id.replace('-', '_')[:8]}"
        
        in_args = []
        for port in self.definition.inputs:
            # Flatten lists if multiple incoming edges to custom port
            v = input_vars.get(port.id, "None")
            if isinstance(v, list):
                in_args.append(f"[{', '.join(v)}]")
            else:
                in_args.append(v)
            
        out_args = []
        for port in self.definition.outputs:
            out_args.append(output_vars.get(port.id, "None"))
            
        in_str = ", ".join(in_args)
        
        if not out_args:
            return f"{var_name}({in_str})"
        elif len(out_args) == 1:
            return f"{out_args[0]} = {var_name}({in_str})"
        else:
            return f"{', '.join(out_args)} = {var_name}({in_str})"
            
    def infer_shapes(self, incoming: dict, params: dict) -> dict:
        return {port.id: ("ANY",) for port in self.definition.outputs}


# ---------------------------------------------------------------------------
# Shape Inference Pass
# ---------------------------------------------------------------------------

def shape_inference_multi_graph(graphs: Dict[str, Any], main_graph_id: str):
    """
    Runs shape inference over all graphs. We return node shapes aggregated by node_id
    across all graphs since node IDs are unique.
    """
    sorted_gids = topological_sort_graphs(graphs)
    all_node_shapes = {}
    all_node_params = {}
    
    for gid in sorted_gids:
        graph_data = graphs[gid]
        sorted_nodes = topological_sort(graph_data.nodes, graph_data.edges)
        ns, np = shape_inference_pass(sorted_nodes, graph_data.edges, graphs)
        all_node_shapes.update(ns)
        all_node_params.update(np)
        
    return all_node_shapes, all_node_params

def shape_inference_pass(
    sorted_nodes: List[Node],
    edges: List[Edge],
    graphs: Dict[str, Any] = None
) -> Tuple[Dict[str, Dict[str, Tuple]], Dict[str, Dict[str, Any]]]:
    graphs = graphs or {}
    node_ids = {node.id for node in sorted_nodes}
    valid_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    tensor_shapes: Dict[str, Tuple] = {}
    node_out_shapes: Dict[str, Dict[str, Tuple]] = {}
    updated_node_params: Dict[str, Dict[str, Any]] = {}

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        
        if block_id == "custom_module":
            dep_id = getattr(node.data, "custom_module_id", "")
            if dep_id in graphs:
                dep_graph = graphs[dep_id]
                class_name = _sanitize(dep_graph.name).capitalize()
                block_def = _get_custom_block_def(dep_id, graphs)
                block = CustomModuleBlock(block_def, class_name)
            else:
                continue
        else:
            block = get_block_by_id(block_id)
            
        if not block:
            continue

        incoming: Dict[str, Any] = {}
        incoming_edge_ids: List[str] = []
        for port in block.definition.inputs:
            matching_edges = [
                e for e in valid_edges if e.target == node.id and e.targetHandle == port.id
            ]
            if port.is_list:
                shapes_list = []
                for e in matching_edges:
                    src_key = f"{e.source}_{e.sourceHandle}"
                    shapes_list.append(tensor_shapes.get(src_key, ("ANY",)))
                    incoming_edge_ids.append(e.id)
                incoming[port.id] = shapes_list if shapes_list else [("ANY",)]
            else:
                if matching_edges:
                    e = matching_edges[0]
                    src_key = f"{e.source}_{e.sourceHandle}"
                    incoming[port.id] = tensor_shapes.get(src_key, ("ANY",))
                    incoming_edge_ids.append(e.id)
                else:
                    incoming[port.id] = ("ANY",)

        original_params = dict(node.data.paramValues)

        try:
            out_shapes = block.infer_shapes(incoming, node.data.paramValues)
        except ValueError as exc:
            raise ShapeError(
                message=str(exc),
                node_id=node.id,
                node_label=node.data.label,
                edge_ids=incoming_edge_ids,
            ) from exc

        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape

        for k, v in node.data.paramValues.items():
            if k not in original_params or original_params[k] != v:
                if node.id not in updated_node_params:
                    updated_node_params[node.id] = {}
                updated_node_params[node.id][k] = v

        combined_shapes = {**incoming, **out_shapes}
        node_out_shapes[node.id] = combined_shapes

    return node_out_shapes, updated_node_params


# ---------------------------------------------------------------------------
# Code Generation Pass
# ---------------------------------------------------------------------------

def _build_input_var_map(sorted_nodes: List[Node]) -> Dict[str, str]:
    INPUT_IDS = {"input", "gourav"}
    input_nodes = [n for n in sorted_nodes if _resolve_block_id(n) in INPUT_IDS]

    label_counts: Dict[str, int] = {}
    var_map: Dict[str, str] = {}

    for node in input_nodes:
        raw = (node.data.varName or "").strip()
        base = _label_to_identifier(raw) if raw else _label_to_identifier(node.data.label)
        base = base or "x"

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


def generate_pytorch_code(graphs: Dict[str, Any], main_graph_id: str) -> str:
    sorted_gids = topological_sort_graphs(graphs)
    
    # Run shape inference pass across all graphs to catch mismatches early
    shape_inference_multi_graph(graphs, main_graph_id)
    
    code = [
        "import torch",
        "import torch.nn as nn",
        ""
    ]
    
    for gid in sorted_gids:
        graph_data = graphs[gid]
        is_main = (gid == main_graph_id)
        class_name = _sanitize(graph_data.name).capitalize() if not is_main else "Model"
        if not class_name:
            class_name = f"Module_{gid[:8]}"
            
        try:
            sorted_nodes = topological_sort(graph_data.nodes, graph_data.edges)
        except ValueError as e:
            raise ValueError(f"Cycle detected in module {graph_data.name}: {e}")
            
        graph_code = _generate_single_graph_code(
            sorted_nodes, graph_data.edges, class_name, graphs
        )
        code.append(graph_code)
        code.append("")
        
    return "\n".join(code).strip()


def _generate_single_graph_code(sorted_nodes: List[Node], edges: List[Edge], class_name: str, graphs: Dict[str, Any]) -> str:
    node_ids = {node.id for node in sorted_nodes}
    valid_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    code = [
        f"class {class_name}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    init_lines: List[str] = []

    var_map = _build_input_var_map(sorted_nodes)

    INPUT_IDS = {"input"}
    forward_arg_names = [
        var_map[f"{n.id}_out"]
        for n in sorted_nodes
        if _resolve_block_id(n) in INPUT_IDS
    ]
    forward_lines: List[str] = [f"    def forward(self, {', '.join(forward_arg_names)}):"]

    hint_counts: Dict[str, int] = {}
    return_vars: List[str] = []

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        
        if block_id in INPUT_IDS:
            continue

        if block_id == "custom_module":
            dep_id = getattr(node.data, "custom_module_id", "")
            if dep_id in graphs:
                dep_graph = graphs[dep_id]
                custom_class_name = _sanitize(dep_graph.name).capitalize()
                block_def = _get_custom_block_def(dep_id, graphs)
                block = CustomModuleBlock(block_def, custom_class_name)
            else:
                forward_lines.append(f"        # WARNING: unknown custom module '{dep_id}' — skipped")
                continue
        else:
            block = get_block_by_id(block_id)
            
        if not block:
            forward_lines.append(f"        # WARNING: unknown block '{block_id}' — skipped")
            continue

        params = node.data.paramValues

        input_vars: Dict[str, Any] = {}
        for port in block.definition.inputs:
            matching_edges = [
                e for e in valid_edges if e.target == node.id and e.targetHandle == port.id
            ]
            if port.is_list:
                v_list = []
                for e in matching_edges:
                    src_key = f"{e.source}_{e.sourceHandle}"
                    v = var_map.get(src_key, "None")
                    if v != "None":
                        v_list.append(v)
                input_vars[port.id] = v_list
            else:
                if matching_edges:
                    src_key = f"{matching_edges[0].source}_{matching_edges[0].sourceHandle}"
                    input_vars[port.id] = var_map.get(src_key, "None")
                else:
                    input_vars[port.id] = "None"

        if block_id == "output":
            in_var = input_vars.get("in", [])
            if isinstance(in_var, list):
                return_vars.extend([v for v in in_var if v != "None"])
            elif in_var != "None":
                return_vars.append(in_var)
            continue

        output_vars: Dict[str, str] = {}
        raw_user_var = (node.data.varName or "").strip()
        user_var = _sanitize(raw_user_var) if raw_user_var else ""
        if user_var and (not user_var[0].isalpha() and user_var[0] != "_"):
            user_var = "x_" + user_var
        
        for port in block.definition.outputs:
            if user_var:
                out_var = user_var if len(block.definition.outputs) == 1 else f"{user_var}_{port.id}"
            else:
                out_var = _build_output_var(port, node.id, params, hint_counts)

            output_vars[port.id] = out_var
            var_map[f"{node.id}_{port.id}"] = out_var

        if not block.definition.is_functional:
            init_code = block.emit_init(node.id, params)
            if init_code:
                init_lines.append(f"        {init_code}")

        forward_code = block.emit_forward(node.id, input_vars, output_vars, params)
        if forward_code:
            forward_lines.append(f"        {forward_code}")

    if return_vars:
        forward_lines.append(f"        return {', '.join(return_vars)}")
    else:
        if len(forward_lines) == 1:
            forward_lines.append("        pass")

    if not init_lines:
        init_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    return "\n".join(code)
