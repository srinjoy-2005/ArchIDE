from models import CompileRequest, Node, Edge, PortDef, BlockDef, NodeData, ParamDef
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
    if block_id == "custom_module":
        return "custom"
    return block_id or "unknown"

def _to_pascal_case(name: str) -> str:
    """Turn a module name or string into a clean PascalCase Python class name."""
    words = [w for w in re.split(r"[^a-zA-Z0-9]+", name) if w]
    if not words:
        return "Model"
    res = ""
    for w in words:
        if w.isupper():
            res += w.capitalize()
        elif any(c.isupper() for c in w[1:]):
            res += w[0].upper() + w[1:]
        else:
            res += w.capitalize()
    return res

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
    def __init__(self, message: str, node_id: str, node_label: str, edge_ids: List[str] | None = None):
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


def _resolve_custom_dep(node: Node, graphs: Dict[str, Any]) -> Optional[str]:
    """Resolves a custom module node to its canonical key in graphs dict."""
    dep_id = getattr(node.data, "custom_module_id", "")
    if dep_id and dep_id in graphs:
        return dep_id
    label = getattr(node.data, "label", "").replace(".arch", "").strip()
    clean_dep = dep_id.replace(".arch", "").split("/")[-1].strip() if dep_id else ""
    for g_path, g_data in graphs.items():
        g_name = getattr(g_data, "name", "")
        g_stem = g_path.replace(".arch", "").split("/")[-1]
        if (
            (dep_id and (dep_id == g_path or dep_id == g_name or dep_id in g_path))
            or (clean_dep and (clean_dep == g_stem or clean_dep == g_name))
            or (label and (label == g_path or label == g_name or label == g_stem or label in g_path or g_stem in label))
        ):
            return g_path
    return None


def topological_sort_graphs(graphs: Dict[str, Any]) -> List[str]:
    adj = {gid: [] for gid in graphs}
    in_degree = {gid: 0 for gid in graphs}
    for gid, data in graphs.items():
        for node in data.nodes:
            block_id = _resolve_block_id(node)
            if block_id == "custom":
                dep_id = _resolve_custom_dep(node, graphs)
                if dep_id and dep_id in graphs and dep_id != gid:
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
        # Fallback to key order if circular or unresolvable
        return list(graphs.keys())
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
    
    # Map init_param variables to ParamDefs
    variables = getattr(graph_data, "variables", []) or []
    params = [
        ParamDef(name=v.name, type=v.type, default=v.default, section="basic")
        for v in variables if v.scope == "init_param"
    ]
    
    # Fallback to legacy parameters if no variables exist
    if not params:
        params = getattr(graph_data, "parameters", []) or []
    
    return BlockDef(
        id=f"custom_module_{gid}",
        name=graph_data.name,
        category="Custom Modules",
        color="#eab308",
        is_functional=False,
        inputs=inputs,
        outputs=outputs,
        params=params
    )

class CustomModuleBlock(BaseBlock):
    def __init__(self, definition: BlockDef, class_name: str, dep_graph: Optional[Any] = None, graphs: Optional[Dict[str, Any]] = None):
        self._definition = definition
        self.class_name = class_name
        self.dep_graph = dep_graph
        self.graphs = graphs or {}
        
    @property
    def definition(self) -> BlockDef:
        return self._definition
        
    def emit_init(self, node_id: str, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        kwargs = []

        def format_val(v: Any) -> str:
            if isinstance(v, str):
                if v.startswith("self."):
                    return v
                if v.isdigit():
                    return v
                try:
                    float(v)
                    return v
                except ValueError:
                    pass
                if v.isidentifier():
                    return v
                return repr(v)
            return str(v)

        if self.definition.params:
            for p in self.definition.params:
                val = params.get(p.name, p.default)
                kwargs.append(f"{p.name}={format_val(val)}")
        elif params:
            for k, v in params.items():
                if k.startswith("_"):
                    continue
                kwargs.append(f"{k}={format_val(v)}")
                    
        args_str = ", ".join(kwargs)
        return f"{var_name} = {self.class_name}({args_str})"
        
    def emit_forward(self, node_id: str, input_vars: dict, output_vars: dict, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        
        in_args = []
        for port_idx, port in enumerate(self.definition.inputs):
            v = input_vars.get(port.id)
            if v is None or v == "None":
                v = input_vars.get(f"in_{port_idx+1}")
            if (v is None or v == "None") and port_idx == 0:
                v = input_vars.get("in")
            if v is None or v == "None":
                vals = [val for val in input_vars.values() if val != "None"]
                v = vals[port_idx] if port_idx < len(vals) else "None"

            if isinstance(v, list):
                in_args.append(f"[{', '.join(v)}]")
            elif v != "None":
                in_args.append(v)
            
        out_args = []
        for port in self.definition.outputs:
            out_args.append(output_vars.get(port.id, "None"))
            
        valid_in = [a for a in in_args if a != "None"]
        in_str = ", ".join(valid_in) if valid_in else "x_input"
        valid_out = [a for a in out_args if a != "None"]
        
        if not valid_out:
            return f"{var_name}({in_str})"
        elif len(valid_out) == 1:
            return f"{valid_out[0]} = {var_name}({in_str})"
        else:
            return f"{', '.join(valid_out)} = {var_name}({in_str})"
            
    def infer_shapes(self, incoming: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        if not self.dep_graph:
            return {port.id: ("ANY",) for port in self.definition.outputs}
            
        try:
            # Create isolated copy of nodes so instance-level parameter auto-inference doesn't mutate graph
            sub_nodes = [
                Node(
                    id=n.id,
                    data=NodeData(
                        block_id=n.data.block_id,
                        label=n.data.label,
                        is_functional=n.data.is_functional,
                        paramValues=dict(n.data.paramValues),
                        varName=n.data.varName,
                        custom_module_id=getattr(n.data, "custom_module_id", ""),
                    ),
                    position=n.position
                )
                for n in self.dep_graph.nodes
            ]
            sorted_nodes = topological_sort(sub_nodes, self.dep_graph.edges)

            # Map incoming port shapes to sub-graph input node IDs
            sub_input_nodes = [n for n in sub_nodes if _resolve_block_id(n) in {"input", "gourav"}]
            mapped_initial_shapes = {}
            for idx, in_node in enumerate(sub_input_nodes):
                if in_node.id in incoming and incoming[in_node.id] != ("ANY",):
                    mapped_initial_shapes[in_node.id] = incoming[in_node.id]
                elif f"in_{idx+1}" in incoming and incoming[f"in_{idx+1}"] != ("ANY",):
                    mapped_initial_shapes[in_node.id] = incoming[f"in_{idx+1}"]
                elif idx == 0 and "in" in incoming and incoming["in"] != ("ANY",):
                    mapped_initial_shapes[in_node.id] = incoming["in"]
                else:
                    in_vals = [v for v in incoming.values() if v != ("ANY",)]
                    if idx < len(in_vals):
                        mapped_initial_shapes[in_node.id] = in_vals[idx]

            sub_shapes, _ = shape_inference_pass(
                sorted_nodes,
                self.dep_graph.edges,
                self.graphs,
                initial_input_shapes=mapped_initial_shapes,
                variables=getattr(self.dep_graph, "variables", []) or []
            )
            
            out_shapes = {}
            for port in self.definition.outputs:
                out_shape_dict = sub_shapes.get(port.id, {})
                shape = out_shape_dict.get("in", ("ANY",))
                if isinstance(shape, list):
                    shape = shape[0] if shape else ("ANY",)
                out_shapes[port.id] = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
            return out_shapes
        except ShapeError as se:
            raise ShapeError(
                message=f"In module '{self.definition.name}': {se}",
                node_id=se.node_id,
                node_label=se.node_label,
                edge_ids=se.edge_ids
            ) from se
        except Exception:
            return {port.id: ("ANY",) for port in self.definition.outputs}


class SequentialModuleBlock(BaseBlock):
    def __init__(self, raw_id: str, label: str = "Sequential"):
        self._raw_id = raw_id
        self._label = label
        self._definition = BlockDef(
            id=f"sequential_{raw_id}",
            name=label,
            category="Custom Modules",
            color="#eab308",
            is_functional=False,
            inputs=[PortDef(id="in", name="Input")],
            outputs=[PortDef(id="out", name="Output")],
            params=[]
        )

    @property
    def definition(self) -> BlockDef:
        return self._definition

    def emit_init(self, node_id: str, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        indexed_items = []
        param_items = []
        for k, v in params.items():
            if k.startswith("param_") and k[6:].isdigit():
                indexed_items.append((int(k[6:]), str(v)))
            elif not k.startswith("_"):
                param_items.append(str(v))
        if indexed_items:
            indexed_items.sort(key=lambda x: x[0])
            layers = [item[1] for item in indexed_items]
        else:
            layers = param_items

        if not layers:
            return f"{var_name} = nn.Sequential()"

        layers_str = ",\n            ".join(layers)
        return f"{var_name} = nn.Sequential(\n            {layers_str}\n        )"

    def emit_forward(self, node_id: str, input_vars: dict, output_vars: dict, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        in_var = input_vars.get("in", "x_input")
        if isinstance(in_var, list):
            in_var = in_var[0] if in_var else "x_input"
        out_var = output_vars.get("out", "x")
        return f"{out_var} = {var_name}({in_var})"

    def infer_shapes(self, incoming: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {"out": ("ANY",)}


class GenericCustomModuleBlock(BaseBlock):
    def __init__(self, class_name: str, input_ports: Optional[List[PortDef]] = None, output_ports: Optional[List[PortDef]] = None):
        self.class_name = class_name
        inputs = input_ports or [PortDef(id="in", name="Input")]
        outputs = output_ports or [PortDef(id="out", name="Output")]
        self._definition = BlockDef(
            id=f"generic_{class_name.lower()}",
            name=class_name,
            category="Custom Modules",
            color="#eab308",
            is_functional=False,
            inputs=inputs,
            outputs=outputs,
            params=[]
        )

    @property
    def definition(self) -> BlockDef:
        return self._definition

    def emit_init(self, node_id: str, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        args = []
        indexed_items = []
        for k, v in params.items():
            if k.startswith("param_") and k[6:].isdigit():
                indexed_items.append((int(k[6:]), str(v)))
            elif not k.startswith("_"):
                val_str = str(v)
                if isinstance(v, str) and not v.startswith("self.") and not v.isdigit():
                    try:
                        float(v)
                    except ValueError:
                        if not v.isidentifier():
                            val_str = repr(v)
                args.append(f"{k}={val_str}")
        if indexed_items:
            indexed_items.sort(key=lambda x: x[0])
            for _, val in indexed_items:
                args.append(val)

        args_str = ", ".join(args)
        return f"{var_name} = {self.class_name}({args_str})"

    def emit_forward(self, node_id: str, input_vars: dict, output_vars: dict, params: dict, member_name: Optional[str] = None) -> str:
        var_name = member_name or f"self.custom_{node_id.replace('-', '_')}"
        in_args = []
        for p in self._definition.inputs:
            v = input_vars.get(p.id, "None")
            if isinstance(v, list):
                in_args.append(f"[{', '.join(v)}]")
            else:
                in_args.append(v)
        valid_in = [a for a in in_args if a != "None"]
        in_str = ", ".join(valid_in) if valid_in else "x_input"
        out_args = [output_vars.get(p.id, "x") for p in self._definition.outputs]
        if not out_args:
            return f"{var_name}({in_str})"
        elif len(out_args) == 1:
            return f"{out_args[0]} = {var_name}({in_str})"
        else:
            return f"{', '.join(out_args)} = {var_name}({in_str})"

    def infer_shapes(self, incoming: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Tuple]:
        return {p.id: ("ANY",) for p in self._definition.outputs}


def _resolve_node_block(node: Node, graphs: Dict[str, Any], edges: Optional[List[Edge]] = None) -> Optional[BaseBlock]:
    block_id = _resolve_block_id(node)
    if block_id == "custom":
        dep_id = _resolve_custom_dep(node, graphs)
        if dep_id and dep_id in graphs:
            dep_graph = graphs[dep_id]
            custom_class_name = _to_pascal_case(dep_graph.name)
            block_def = _get_custom_block_def(dep_id, graphs)
            return CustomModuleBlock(block_def, custom_class_name, dep_graph=dep_graph, graphs=graphs)

        raw_dep = getattr(node.data, "custom_module_id", "") or getattr(node.data, "label", "") or "Custom"
        raw_dep_lower = raw_dep.lower()
        if raw_dep_lower in ("sequential", "nn.sequential", "torch.nn.sequential") or "sequential" in getattr(node.data, "label", "").lower():
            return SequentialModuleBlock(node.id, label=getattr(node.data, "label", "Sequential"))

        class_name = _to_pascal_case(raw_dep) if not raw_dep.isupper() else raw_dep
        if getattr(node.data, "label", "") and getattr(node.data, "label") not in ("Custom", "custom_module"):
            class_name = _to_pascal_case(node.data.label)

        input_ports: List[PortDef] = []
        raw_inputs = getattr(node.data, "inputs", []) or []
        for p in raw_inputs:
            if isinstance(p, dict) and "id" in p:
                input_ports.append(PortDef(id=p["id"], name=p.get("name", "Input")))
            elif hasattr(p, "id"):
                input_ports.append(PortDef(id=p.id, name=getattr(p, "name", "Input")))

        if edges:
            node_target_edges = [e for e in edges if e.target == node.id]
            existing_port_ids = {p.id for p in input_ports}
            for idx, e in enumerate(node_target_edges):
                target_handle = e.targetHandle or ("in" if idx == 0 else f"in_{idx + 1}")
                if target_handle not in existing_port_ids:
                    input_ports.append(PortDef(id=target_handle, name=f"Input {idx + 1}"))
                    existing_port_ids.add(target_handle)

        output_ports: List[PortDef] = []
        raw_outputs = getattr(node.data, "outputs", []) or []
        for p in raw_outputs:
            if isinstance(p, dict) and "id" in p:
                output_ports.append(PortDef(id=p["id"], name=p.get("name", "Output")))
            elif hasattr(p, "id"):
                output_ports.append(PortDef(id=p.id, name=getattr(p, "name", "Output")))

        return GenericCustomModuleBlock(
            class_name,
            input_ports=input_ports or [PortDef(id="in", name="Input")],
            output_ports=output_ports or [PortDef(id="out", name="Output")]
        )

    return get_block_by_id(block_id)


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
        working_nodes = [
            Node(
                id=n.id,
                data=NodeData(
                    block_id=n.data.block_id,
                    label=n.data.label,
                    is_functional=n.data.is_functional,
                    paramValues=dict(n.data.paramValues),
                    varName=n.data.varName,
                    custom_module_id=getattr(n.data, "custom_module_id", ""),
                ),
                position=n.position
            )
            for n in graph_data.nodes
        ]
        sorted_nodes = topological_sort(working_nodes, graph_data.edges)
        variables = getattr(graph_data, "variables", []) or []
        ns, np = shape_inference_pass(sorted_nodes, graph_data.edges, graphs, variables=variables)
        all_node_shapes.update(ns)
        all_node_params.update(np)
        
        # Apply inferred parameters back to the main graph only
        if gid == main_graph_id:
            for n in graph_data.nodes:
                if n.id in np:
                    n.data.paramValues.update(np[n.id])
        
    return all_node_shapes, all_node_params

def shape_inference_pass(
    sorted_nodes: List[Node],
    edges: List[Edge],
    graphs: Dict[str, Any] | None = None,
    initial_input_shapes: Dict[str, Tuple] | None = None,
    variables: List[Any] | None = None
) -> Tuple[Dict[str, Dict[str, Tuple]], Dict[str, Dict[str, Any]]]:
    graphs = graphs or {}
    node_ids = {node.id for node in sorted_nodes}
    valid_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    tensor_shapes: Dict[str, Tuple] = {}
    node_out_shapes: Dict[str, Dict[str, Tuple]] = {}
    updated_node_params: Dict[str, Dict[str, Any]] = {}

    for node in sorted_nodes:
        block_id = node.data.block_id
        block = _resolve_node_block(node, graphs, valid_edges)
        if not block:
            continue

        incoming: Dict[str, Any] = {}
        incoming_edge_ids: List[str] = []
        for port_idx, port in enumerate(block.definition.inputs):
            matching_edges = [
                e for e in valid_edges if e.target == node.id and e.targetHandle == port.id
            ]
            if not matching_edges:
                candidate_handles = [port.id, f"in_{port_idx+1}", "in" if port_idx == 0 else f"in_{port_idx+1}"]
                matching_edges = [
                    e for e in valid_edges if e.target == node.id and (e.targetHandle in candidate_handles or (len(block.definition.inputs) == 1 and e.targetHandle in ("in", "input", "")))
                ]
                if not matching_edges and block_id == "custom":
                    all_target_edges = [e for e in valid_edges if e.target == node.id]
                    if port_idx < len(all_target_edges):
                        matching_edges = [all_target_edges[port_idx]]

            if port.is_list:
                shapes_list = []
                for e in matching_edges:
                    src_key = f"{e.source}_{e.sourceHandle}"
                    shape = tensor_shapes.get(src_key) or tensor_shapes.get(f"{e.source}_out") or ("ANY",)
                    shapes_list.append(shape)
                    incoming_edge_ids.append(e.id)
                incoming[port.id] = shapes_list if shapes_list else [("ANY",)]
            else:
                if matching_edges:
                    e = matching_edges[0]
                    src_key = f"{e.source}_{e.sourceHandle}"
                    shape = tensor_shapes.get(src_key) or tensor_shapes.get(f"{e.source}_out") or ("ANY",)
                    incoming[port.id] = shape
                    incoming_edge_ids.append(e.id)
                else:
                    incoming[port.id] = ("ANY",)

        original_params = dict(node.data.paramValues)

        # Dual-Evaluation: substitute variable defaults to compute shapes
        params_for_inference = dict(node.data.paramValues)
        if variables:
            var_map = {v.name: v for v in variables}
            var_defaults = {v.name: v.default for v in variables}
            # Build a param-type lookup from the block definition
            param_type_map = {p.name: p.type for p in block.definition.params}
            for k, v in list(params_for_inference.items()):
                if isinstance(v, str) and "@var:" in v:
                    try:
                        parsed = v
                        for v_name, v_def in var_defaults.items():
                            parsed = re.sub(fr'@var:{v_name}\b', str(v_def), parsed)
                        # Evaluate basic math if it looks purely numeric
                        if re.match(r'^[0-9+\-*/().\s]+$', parsed):
                            params_for_inference[k] = eval(parsed, {"__builtins__": None}, {})
                        else:
                            # Write back the substituted string as-is (e.g. shape = "(22,224)")
                            params_for_inference[k] = parsed
                    except Exception:
                        pass # Ignore and pass raw string if it fails
                elif isinstance(v, (list, tuple)):
                    substituted = []
                    for item in v:
                        if isinstance(item, str) and "@var:" in item:
                            parsed_item = item
                            for v_name, v_def in var_defaults.items():
                                parsed_item = re.sub(fr'@var:{v_name}\b', str(v_def), parsed_item)
                            try:
                                if re.match(r'^[0-9+\-*/().\s]+$', parsed_item):
                                    item = eval(parsed_item, {"__builtins__": None}, {})
                                else:
                                    item = parsed_item
                            except Exception:
                                pass
                        substituted.append(item)
                    params_for_inference[k] = tuple(substituted) if isinstance(v, tuple) else substituted
        
        # Handle input node override if provided via initial_input_shapes
        if block_id in {"input", "gourav"} and initial_input_shapes and node.id in initial_input_shapes:
            override_shape = initial_input_shapes[node.id]
            if override_shape and override_shape != ("ANY",):
                out_shapes = {"out": override_shape}
            else:
                out_shapes = block.infer_shapes(incoming, params_for_inference)
        else:
            try:
                out_shapes = block.infer_shapes(incoming, params_for_inference)
            except ValueError as exc:
                raise ShapeError(
                    message=str(exc),
                    node_id=node.id,
                    node_label=node.data.label,
                    edge_ids=incoming_edge_ids,
                ) from exc

        for port_id, shape in out_shapes.items():
            tensor_shapes[f"{node.id}_{port_id}"] = shape

        for k, v in params_for_inference.items():
            if k in original_params and original_params[k] in (-1, "?", "") and v not in (-1, "?", ""):
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
            # Enforce x_ prefix for valid Python identifiers and PyTorch tensor conventions
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


def generate_pytorch_code(graphs: Dict[str, Any], main_graph_id: str, file_paths: Dict[str, str]|None = None) -> Tuple[Dict[str, str], Dict, Dict]:
    if file_paths is None:
        file_paths = {}
    
    sorted_gids = topological_sort_graphs(graphs)
    all_node_shapes, all_node_params = shape_inference_multi_graph(graphs, main_graph_id)
    
    files = {}
    
    for gid in sorted_gids:
        graph_data = graphs[gid]
        is_main = (gid == main_graph_id)
        class_name = _to_pascal_case(graph_data.name) if not is_main else "Model"
        if not class_name:
            class_name = f"Module_{gid[:8]}"
            
        try:
            sorted_nodes = topological_sort(graph_data.nodes, graph_data.edges)
        except ValueError as e:
            raise ValueError(f"Cycle detected in module {graph_data.name}: {e}")
            
        # Collect custom module dependencies for imports
        custom_deps = set()
        for node in sorted_nodes:
            if _resolve_block_id(node) == "custom":
                dep_id = _resolve_custom_dep(node, graphs)
                if dep_id and dep_id in graphs:
                    custom_deps.add(dep_id)
                    
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import math",
        ]
        
        for dep_id in custom_deps:
            dep_path = file_paths.get(dep_id, dep_id)
            dep_graph = graphs[dep_id]
            dep_class = _to_pascal_case(dep_graph.name)
            clean_mod = dep_path.replace(".arch", "").replace(".json", "").strip("/").replace("/", ".")
            imports.append(f"from {clean_mod} import {dep_class}")

        imports.append("")

        graph_code = _generate_single_graph_code(
            sorted_nodes,
            graph_data.edges,
            class_name,
            graphs,
            graph_data=graph_data,
            inferred_params=all_node_params
        )

        # Emit local_const variables as module-level constants before the class
        local_consts = [v for v in getattr(graph_data, "variables", []) if v.scope == "local_const"]
        const_lines = [f"{v.name.upper()} = {repr(v.default) if isinstance(v.default, str) else v.default}" for v in local_consts]
        const_block = "\n".join(const_lines) + "\n\n" if const_lines else ""

        file_code = "\n".join(imports) + "\n" + const_block + graph_code
        files[gid] = file_code
        
    return files, all_node_shapes, all_node_params


def _generate_single_graph_code(
    sorted_nodes: List[Node],
    edges: List[Edge],
    class_name: str,
    graphs: Dict[str, Any],
    graph_data: Any = None,
    inferred_params: Dict[str, Dict[str, Any]]|None = None
) -> str:
    node_ids = {node.id for node in sorted_nodes}
    valid_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

    # ── Variable resolution ──────────────────────────────────────────────────
    # Prefer the new `variables` list; fall back to legacy `parameters` for old graphs.
    variables = getattr(graph_data, "variables", []) if graph_data else []
    params_decl = getattr(graph_data, "parameters", []) if graph_data else []

    init_params  = [v for v in variables if v.scope == "init_param"]
    local_consts = [v for v in variables if v.scope == "local_const"]

    # If no new-style variables but legacy parameters exist, treat them as init_params
    if not variables and params_decl:
        init_params = params_decl   # ParamDef has same name/type/default shape

    # Build __init__ signature
    if init_params:
        init_args = ["self"]
        for v in init_params:
            type_map = {"string": "str", "int": "int", "float": "float", "bool": "bool", "shape": "tuple", "tuple": "tuple"}
            py_type = type_map.get(v.type, v.type)
            t_str = f": {py_type}" if py_type else ""
            d_val = repr(v.default) if isinstance(v.default, str) else str(v.default)
            init_args.append(f"{v.name}{t_str} = {d_val}")
        init_sig = f"    def __init__({', '.join(init_args)}):"
    else:
        init_sig = "    def __init__(self):"

    code = [
        f"class {class_name}(nn.Module):",
        init_sig,
        "        super().__init__()",
    ]
    # Store init_params as self.* so they can be used in forward()
    for v in init_params:
        code.append(f"        self.{v.name} = {v.name}")

    init_lines: List[str] = []

    var_map = _build_input_var_map(sorted_nodes)

    INPUT_IDS = {"input"}
    forward_arg_names = [
        var_map[f"{n.id}_out"]
        for n in sorted_nodes
        if _resolve_block_id(n) in INPUT_IDS
    ]
    forward_lines: List[str] = [
        f"    def forward(self, {', '.join(forward_arg_names)}):",
        "        # Note: All input variables and un-aliased custom variables are prefixed with 'x_'",
        "        # to ensure valid Python identifiers and adhere to PyTorch conventions."
    ]

    hint_counts: Dict[str, int] = {}
    return_vars: List[str] = []

    used_member_names: Set[str] = set()
    node_member_map: Dict[str, str] = {}

    for node in sorted_nodes:
        block_id = _resolve_block_id(node)
        
        if block_id in INPUT_IDS:
            continue

        block = _resolve_node_block(node, graphs, valid_edges)
        if not block:
            forward_lines.append(f"        # WARNING: unknown block '{block_id}' — skipped")
            continue

        if not block.definition.is_functional:
            raw_lbl = getattr(node.data, "label", "") or ""
            clean_lbl = _sanitize(raw_lbl) if raw_lbl and raw_lbl.lower() not in ("custom", "custom_module", "sequential", "input", "output") else ""
            if not clean_lbl and getattr(node.data, "custom_module_id", ""):
                clean_lbl = _sanitize(node.data.custom_module_id.split("/")[-1])
            
            if clean_lbl:
                member_cand = clean_lbl
            elif block_id in ("linear", "conv1d", "conv2d", "embedding", "layernorm", "batchnorm2d", "maxpool2d", "avgpool2d", "adaptiveavgpool2d", "dropout", "relu", "gelu", "silu", "sigmoid", "tanh", "softmax"):
                member_cand = f"layer_{_sanitize(raw_lbl or block_id)}"
            else:
                member_cand = f"custom_{node.id.replace('-', '_')}"
            
            if member_cand in used_member_names:
                member_cand = f"{member_cand}_{node.id[:4]}"
            used_member_names.add(member_cand)
            node_member_map[node.id] = f"self.{member_cand}"

        params = dict(node.data.paramValues)

        # 1. Legacy: substitute constructor argument names if parameter is in graph_data.parameters
        if graph_data and getattr(graph_data, "parameters", []):
            for p in graph_data.parameters:
                if params.get(p.name) in (-1, None, ""):
                    params[p.name] = p.name

        # 2. Fill any remaining -1 parameters from inferred_params
        if inferred_params and node.id in inferred_params:
            for k, v in inferred_params[node.id].items():
                if params.get(k) == "LAZY" and block_id in ("linear", "conv1d", "conv2d"):
                    continue
                if params.get(k) in (-1, "?", "", "LAZY"):
                    params[k] = v

        # 3. Resolve @var: bindings — replace with self.<name> for init_params
        #    or <NAME> constant for local_consts.
        var_name_to_scope = {v.name: v.scope for v in variables}
        
        def replace_var(match):
            var_name = match.group(1)
            scope = var_name_to_scope.get(var_name, "init_param")
            return f"self.{var_name}" if scope == "init_param" else var_name.upper()
            
        for k, val in list(params.items()):
            if isinstance(val, str) and "@var:" in val:
                params[k] = re.sub(r'@var:([a-zA-Z0-9_]+)', replace_var, val)
            elif isinstance(val, (list, tuple)):
                new_seq = []
                for item in val:
                    if isinstance(item, str) and "@var:" in item:
                        new_seq.append(re.sub(r'@var:([a-zA-Z0-9_]+)', replace_var, item))
                    else:
                        new_seq.append(item)
                params[k] = tuple(new_seq) if isinstance(val, tuple) else new_seq
                    
        # 3. Handle inputs
        input_vars: Dict[str, Any] = {}
        for port_idx, port in enumerate(block.definition.inputs):
            matching_edges = [
                e for e in valid_edges if e.target == node.id and e.targetHandle == port.id
            ]
            # Fallback for custom modules or positional handles if exact port.id wasn't matched
            if not matching_edges:
                candidate_handles = [port.id, f"in_{port_idx+1}", "in" if port_idx == 0 else f"in_{port_idx+1}"]
                matching_edges = [
                    e for e in valid_edges if e.target == node.id and (e.targetHandle in candidate_handles or (len(block.definition.inputs) == 1 and e.targetHandle in ("in", "input", "")))
                ]
                if not matching_edges and block_id == "custom":
                    all_target_edges = [e for e in valid_edges if e.target == node.id]
                    if port_idx < len(all_target_edges):
                        matching_edges = [all_target_edges[port_idx]]

            if port.is_list:
                v_list = []
                for e in matching_edges:
                    src_key = f"{e.source}_{e.sourceHandle}"
                    v = var_map.get(src_key) or var_map.get(f"{e.source}_out") or "None"
                    if v != "None":
                        v_list.append(v)
                input_vars[port.id] = v_list
            else:
                if matching_edges:
                    src_key = f"{matching_edges[0].source}_{matching_edges[0].sourceHandle}"
                    input_vars[port.id] = var_map.get(src_key) or var_map.get(f"{matching_edges[0].source}_out") or "None"
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
            # Provide generic out handle fallback
            var_map[f"{node.id}_out"] = out_var

        if not block.definition.is_functional:
            try:
                init_code = block.emit_init(node.id, params, member_name=node_member_map.get(node.id))
            except TypeError:
                init_code = block.emit_init(node.id, params)
            if init_code:
                init_lines.append(f"        {init_code}")

        try:
            forward_code = block.emit_forward(node.id, input_vars, output_vars, params, member_name=node_member_map.get(node.id))
        except TypeError:
            forward_code = block.emit_forward(node.id, input_vars, output_vars, params)
        if forward_code:
            forward_lines.append(f"        {forward_code}")

    if return_vars:
        forward_lines.append(f"        return {', '.join(return_vars)}")
    else:
        if len(forward_lines) <= 3:
            if forward_arg_names:
                forward_lines.append(f"        return {forward_arg_names[0]}")
            else:
                forward_lines.append("        pass")

    if not init_lines and len(init_params) == 0:
        init_lines.append("        pass")

    code.extend(init_lines)
    code.append("")
    code.extend(forward_lines)
    return "\n".join(code)
