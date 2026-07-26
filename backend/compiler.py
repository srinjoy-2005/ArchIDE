from models import CompileRequest, Node, Edge
from typing import List, Dict

def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
    # 1. Build adjacency list and in-degree map
    adj = {node.id: [] for node in nodes}
    in_degree = {node.id: 0 for node in nodes}
    node_map = {node.id: node for node in nodes}

    for edge in edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    # 2. Queue for nodes with no incoming edges
    queue = [node_id for node_id in in_degree if in_degree[node_id] == 0]
    
    sorted_nodes = []
    
    # 3. Kahn's Algorithm
    while queue:
        curr_id = queue.pop(0)
        sorted_nodes.append(node_map[curr_id])
        
        for neighbor in adj[curr_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles (e.g. if the user drew a loop)
    if len(sorted_nodes) != len(nodes):
        raise ValueError("Cycle detected in graph! Cannot compile.")
        
    return sorted_nodes

def generate_pytorch_code(sorted_nodes: List[Node], edges: List[Edge]) -> str:
    # 1. Boilerplate
    code = [
        "import torch",
        "import torch.nn as nn",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()"
    ]
    init_lines = []
    
    # Find all Input blocks and generate forward arguments based on their labels
    input_nodes = [n for n in sorted_nodes if getattr(n.data, 'block_id', '') == "input" or n.data.label == "Input"]
    forward_args = ["self"] + [f"x_{n.data.label.replace(' ', '_').lower()}" for n in input_nodes]
    forward_lines = [f"    def forward({', '.join(forward_args)}):"]
    
    # Track variable names: port_id -> var_name
    # Edge lookup: target_node_id + target_handle -> source_node_id + source_handle
    input_to_source = {}
    for edge in edges:
        target_key = f"{edge.target}_{edge.targetHandle}"
        source_key = f"{edge.source}_{edge.sourceHandle}"
        input_to_source[target_key] = source_key
        
    var_map = {} # Maps source_key to actual variable name (e.g. "x_node_1")
    
    for node in sorted_nodes:
        node_id = node.id
        label = node.data.label
        block_id = getattr(node.data, 'block_id', '')
        params = node.data.paramValues
        
        # Generate variable name for this node's output
        out_var = f"x_{node_id.replace('-', '_')}"
        
        def get_in_var(port_name: str) -> str:
            src_key = input_to_source.get(f"{node_id}_{port_name}")
            return var_map.get(src_key, "None") if src_key else "None"

        # --- Handle Special Nodes ---
        if block_id == "input" or (not block_id and label == "Input"):
            # Map this input's output handle to its function argument name
            arg_name = f"x_{label.replace(' ', '_').lower()}"
            var_map[f"{node_id}_out"] = arg_name
            continue
            
        if block_id == "output" or (not block_id and label == "Output"):
            forward_lines.append(f"        return {get_in_var('in')}")
            continue

        # --- Handle Stateful Layers (in __init__) ---
        if not node.data.is_functional:
            layer_name = f"self.layer_{node_id.replace('-', '_')}"
            
            if block_id == "linear" or (not block_id and label == "Linear"):
                in_feat = params.get("in_features", 128)
                out_feat = params.get("out_features", 64)
                init_lines.append(f"        {layer_name} = nn.Linear({in_feat}, {out_feat})")
            elif block_id == "conv2d" or (not block_id and label == "Conv2D"):
                in_ch = params.get("in_channels", 3)
                out_ch = params.get("out_channels", 16)
                k_size = params.get("kernel_size", 3)
                init_lines.append(f"        {layer_name} = nn.Conv2d({in_ch}, {out_ch}, {k_size})")
            elif block_id == "relu" or (not block_id and label == "ReLU"):
                init_lines.append(f"        {layer_name} = nn.ReLU()")
            elif block_id == "softmax" or (not block_id and label == "Softmax"):
                dim = params.get("dim", 1)
                init_lines.append(f"        {layer_name} = nn.Softmax(dim={dim})")
                
            # Forward pass string
            forward_lines.append(f"        {out_var} = {layer_name}({get_in_var('in')})")
            var_map[f"{node_id}_out"] = out_var
            
        # --- Handle Functional Math Ops (only in forward) ---
        else:
            if block_id == "add" or (not block_id and label == "Add"):
                # Find all edges pointing to this node
                node_incoming = [e for e in edges if e.target == node_id]
                src_vars = [var_map.get(f"{e.source}_{e.sourceHandle}", "None") for e in node_incoming]
                
                if src_vars:
                    add_expr = " + ".join(src_vars)
                    forward_lines.append(f"        {out_var} = {add_expr}")
                else:
                    forward_lines.append(f"        {out_var} = 0")
                var_map[f"{node_id}_out"] = out_var
                
            elif block_id == "sub" or (not block_id and label == "Subtract"):
                a = get_in_var("in_a")
                b = get_in_var("in_b")
                forward_lines.append(f"        {out_var} = {a} - {b}")
                var_map[f"{node_id}_out"] = out_var
                
            elif block_id == "split" or (not block_id and label == "Split (Chunk)"):
                chunks = params.get("chunks", 2)
                dim = params.get("dim", -1)
                
                # Split returns multiple outputs: x_out_1, x_out_2
                out_vars = [f"{out_var}_{i}" for i in range(1, chunks + 1)]
                out_vars_str = ", ".join(out_vars)
                forward_lines.append(f"        {out_vars_str} = torch.chunk({get_in_var('in')}, chunks={chunks}, dim={dim})")
                
                # Map each output handle individually (out_1, out_2, etc.)
                for i in range(1, chunks + 1):
                    var_map[f"{node_id}_out_{i}"] = out_vars[i-1]
            
            # TODO: Other functional ops (mul, div, sin, cos)
            
    if not init_lines:
        init_lines.append("        pass")
        
    if len(forward_lines) == 1:
        forward_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    
    return "\n".join(code)
