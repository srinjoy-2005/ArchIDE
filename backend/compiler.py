from models import CompileRequest, Node, Edge
from typing import List, Dict, Tuple, Any
from blocks import get_block_by_id

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

def shape_inference_pass(sorted_nodes: List[Node], edges: List[Edge]) -> Dict[str, Tuple]:
    # Maps edge keys like `nodeId_portId` to tensor shape tuple
    tensor_shapes: Dict[str, Tuple] = {}
    
    for node in sorted_nodes:
        block_id = getattr(node.data, 'block_id', '').lower()
        if not block_id:
            block_id = node.data.label.split()[0].lower() # fallback
            
        block = get_block_by_id(block_id)
        if not block:
            continue
            
        # Gather incoming shapes
        incoming_shapes = {}
        for port in block.definition.inputs:
            # Find edge targeting this node and port
            incoming_edge = next((e for e in edges if e.target == node.id and e.targetHandle == port.id), None)
            if incoming_edge:
                source_key = f"{incoming_edge.source}_{incoming_edge.sourceHandle}"
                incoming_shapes[port.id] = tensor_shapes.get(source_key, ("ANY",))
            else:
                incoming_shapes[port.id] = ("ANY",)
                
        # Perform shape inference
        out_shapes = block.infer_shapes(incoming_shapes, node.data.paramValues)
        
        # Assign to outgoing keys
        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape
            
    return tensor_shapes


def generate_pytorch_code(sorted_nodes: List[Node], edges: List[Edge]) -> str:
    # Optional: run shape pass
    # shape_inference_pass(sorted_nodes, edges)

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
        block_id = getattr(node.data, 'block_id', '').lower()
        if not block_id:
            if "Split" in label: block_id = "split"
            else: block_id = label.lower()
            
        block = get_block_by_id(block_id)
        if not block:
            continue
            
        params = node.data.paramValues
        
        # Prepare inputs for code generation
        input_vars = {}
        for port in block.definition.inputs:
            src_key = input_to_source.get(f"{node_id}_{port.id}")
            input_vars[port.id] = var_map.get(src_key, "None")
            
        # Prepare outputs for code generation
        output_vars = {}
        for port in block.definition.outputs:
            out_var = f"x_{node_id.replace('-', '_')}"
            if len(block.definition.outputs) > 1:
                # Handle cases like split with multiple outputs
                # E.g., out_1, out_2. We can just use the port id as suffix
                out_var = f"{out_var}_{port.id}"
            output_vars[port.id] = out_var
            var_map[f"{node_id}_{port.id}"] = out_var
            
        # Special case for input variables mapped to function args
        if block_id == "input" or block_id == "gourav":
            arg_name = f"x_{label.replace(' ', '_').lower()}"
            var_map[f"{node_id}_out"] = arg_name
            continue
            
        # Delegate initialization to block
        if not block.definition.is_functional:
            init_code = block.emit_init(node_id, params)
            if init_code:
                init_lines.append(f"        {init_code}")
                
        # Delegate forward execution to block
        forward_code = block.emit_forward(node_id, input_vars, output_vars, params)
        if forward_code:
            forward_lines.append(f"        {forward_code}")
            
    if not init_lines:
        init_lines.append("        pass")
        
    if len(forward_lines) == 1:
        forward_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    
    return "\n".join(code)
