import argparse
import json
import os
import sys
import uuid
from typing import Dict, Any, List, Tuple, Optional

# Ensure we can import from backend
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from blocks import get_block_by_id
from models import Node, Edge, NodeData, GraphData, ArchVariableModel, ParamDef
from compiler import (
    shape_inference_multi_graph,
    generate_pytorch_code,
    ShapeError,
    topological_sort,
)

def generate_id(prefix: str = "") -> str:
    short_uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_uid}" if prefix else short_uid

class AgentGraphCompiler:
    """
    Compiles an Agentic Graph Intermediate Representation (IR) into
    a complete, UI-compatible ArchIDE .arch (React Flow) JSON structure.
    """

    def __init__(self, ir_payload: Dict[str, Any], workspace_dir: Optional[str] = None):
        self.ir = ir_payload
        self.name = self.ir.get("name", "Generated Graph")
        self.workspace_dir = workspace_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../workspace/graphs")
        )

        # Support both 'variables' and legacy 'hyperparameters' / 'parameters'
        self.variables = self.ir.get("variables", [])
        if not self.variables and "hyperparameters" in self.ir:
            self.variables = self.ir.get("hyperparameters", [])
        elif not self.variables and "parameters" in self.ir:
            self.variables = self.ir.get("parameters", [])

        self.ir_nodes = self.ir.get("nodes", {})
        self.ir_edges = self.ir.get("edges", [])

        self.node_ids: Dict[str, str] = {}  # alias -> real UUID
        self.nodes_data: Dict[str, Dict[str, Any]] = {}
        self.edges_data: List[Dict[str, Any]] = []

    def _assign_layers(self) -> Dict[str, int]:
        """Calculates topological layers for layout."""
        adj: Dict[str, List[str]] = {alias: [] for alias in self.ir_nodes}
        in_degree: Dict[str, int] = {alias: 0 for alias in self.ir_nodes}

        for edge_item in self.ir_edges:
            src_alias, dst_alias = self._parse_edge_aliases(edge_item)
            if src_alias and dst_alias and src_alias in adj and dst_alias in in_degree:
                adj[src_alias].append(dst_alias)
                in_degree[dst_alias] += 1

        # Topological sort with layering
        layers: Dict[str, int] = {}
        queue = [n for n in self.ir_nodes if in_degree[n] == 0]
        for n in queue:
            layers[n] = 0

        while queue:
            curr = queue.pop(0)
            curr_layer = layers[curr]
            for neighbor in adj[curr]:
                layers[neighbor] = max(layers.get(neighbor, 0), curr_layer + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Fallback for cycles or disconnected components
        for alias in self.ir_nodes:
            if alias not in layers:
                layers[alias] = 0

        return layers

    def _parse_edge_full(self, edge_item: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extracts (src_alias, src_handle, dst_alias, dst_handle) from edge item."""
        if isinstance(edge_item, str):
            try:
                src_part, dst_part = [p.strip() for p in edge_item.split("->")]
                s_a, s_h = [p.strip() for p in src_part.split(".")]
                d_a, d_h = [p.strip() for p in dst_part.split(".")]
                return s_a, s_h, d_a, d_h
            except Exception:
                return None, None, None, None
        elif isinstance(edge_item, dict):
            s_a = edge_item.get("source_alias") or (edge_item.get("source", "").split(".")[0] if "." in edge_item.get("source", "") else edge_item.get("source"))
            s_h = edge_item.get("source_handle") or (edge_item.get("source", "").split(".")[1] if "." in edge_item.get("source", "") else "out")
            d_a = edge_item.get("target_alias") or (edge_item.get("target", "").split(".")[0] if "." in edge_item.get("target", "") else edge_item.get("target"))
            d_h = edge_item.get("target_handle") or (edge_item.get("target", "").split(".")[1] if "." in edge_item.get("target", "") else "in")
            return s_a, s_h, d_a, d_h
        return None, None, None, None

    def _parse_edge_aliases(self, edge_item: Any) -> Tuple[Optional[str], Optional[str]]:
        """Extracts source and target aliases from string 'src.port -> dst.port' or dict."""
        s_a, _, d_a, _ = self._parse_edge_full(edge_item)
        return s_a, d_a

    def _calculate_layout(self) -> Dict[str, Tuple[float, float]]:
        """Assigns non-overlapping (x, y) coordinates based on topological layers and branch offsets."""
        layers = self._assign_layers()
        layer_groups: Dict[int, List[str]] = {}
        for alias, layer_idx in layers.items():
            layer_groups.setdefault(layer_idx, []).append(alias)

        # Detect branch paths / skip connections to introduce vertical offsets for branch nodes
        skip_intermediates = set()
        adj: Dict[str, Set[str]] = {alias: set() for alias in self.ir_nodes}
        for edge_item in self.ir_edges:
            s_a, _, d_a, _ = self._parse_edge_full(edge_item)
            if s_a and d_a and s_a in adj:
                adj[s_a].add(d_a)

        for src, targets in adj.items():
            for t1 in targets:
                for t2 in targets:
                    if t1 != t2 and t2 in adj.get(t1, set()):
                        skip_intermediates.add(t1)

        coords: Dict[str, Tuple[float, float]] = {}
        X_SPACING = 300
        Y_SPACING = 150

        for layer_idx, aliases in layer_groups.items():
            x = layer_idx * X_SPACING
            total_height = (len(aliases) - 1) * Y_SPACING
            start_y = -(total_height / 2)

            for i, alias in enumerate(aliases):
                y = start_y + (i * Y_SPACING)
                if alias in skip_intermediates and len(aliases) == 1:
                    y -= 100
                coords[alias] = (round(x + 100, 1), round(y + 250, 1))

        return coords

    def _load_custom_module_ports(self, custom_module_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Attempts to load inputs/outputs port definitions from a referenced .arch file."""
        if not custom_module_id:
            return [{"id": "in", "name": "Input", "type": "tensor"}], [{"id": "out", "name": "Output", "type": "tensor"}]

        stem = os.path.splitext(os.path.basename(custom_module_id))[0]
        
        # Check candidate locations
        search_dirs = [self.workspace_dir] if self.workspace_dir else []
        if self.workspace_dir:
            search_dirs.append(os.path.dirname(self.workspace_dir))
            search_dirs.append(os.path.join(self.workspace_dir, "modules"))
            search_dirs.append(os.path.join(os.path.dirname(self.workspace_dir), "modules"))
            search_dirs.append(os.path.join(os.path.dirname(self.workspace_dir), "graphs"))
            search_dirs.append(os.path.join(os.path.dirname(self.workspace_dir), "graphs", "modules"))

        candidates = [
            f"{custom_module_id}.arch",
            f"{custom_module_id}.json",
            f"{stem}.arch",
            f"modules/{stem}.arch",
        ]
        
        found_path = None
        for sdir in search_dirs:
            if not sdir or not os.path.exists(sdir):
                continue
            for cand in candidates:
                p = os.path.join(sdir, cand)
                if os.path.isfile(p):
                    found_path = p
                    break
            if found_path:
                break

        if found_path:
            try:
                with open(found_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                inputs, outputs = [], []
                for n in data.get("nodes", []):
                    b_id = n.get("data", {}).get("block_id")
                    if b_id in {"input", "gourav"}:
                        lbl = n.get("data", {}).get("label") or "Input"
                        inputs.append({"id": n.get("id", f"in_{len(inputs)+1}"), "name": lbl, "type": "tensor"})
                    elif b_id == "output":
                        lbl = n.get("data", {}).get("label") or "Output"
                        outputs.append({"id": n.get("id", f"out_{len(outputs)+1}"), "name": lbl, "type": "tensor"})
                if inputs or outputs:
                    return (
                        inputs or [{"id": "in", "name": "Input", "type": "tensor"}],
                        outputs or [{"id": "out", "name": "Output", "type": "tensor"}],
                    )
            except Exception:
                pass

        return [{"id": "in", "name": "Input", "type": "tensor"}], [{"id": "out", "name": "Output", "type": "tensor"}]

    def compile(self) -> Dict[str, Any]:
        """Compiles the Agentic IR into full ArchIDE JSON."""
        coords = self._calculate_layout()

        # 1. Hydrate Nodes
        for alias, node_info in self.ir_nodes.items():
            block_id = (
                node_info.get("block")
                or node_info.get("block_id")
                or node_info.get("type")
                or "custom"
            )
            custom_module_id = (
                node_info.get("custom_module_id")
                or node_info.get("module")
                or node_info.get("submodule")
                or ""
            )
            if custom_module_id and block_id not in {"custom", "custom_module"}:
                block_id = "custom"

            params = dict(node_info.get("params", {}))
            label = node_info.get("label") or alias.capitalize()
            var_name = node_info.get("var_name") or node_info.get("varName") or ""

            real_id = generate_id(block_id)
            self.node_ids[alias] = real_id
            x, y = coords.get(alias, (100.0, 100.0))

            node_payload: Dict[str, Any] = {
                "id": real_id,
                "type": "custom",
                "position": {"x": x, "y": y},
                "data": {
                    "block_id": block_id,
                    "label": label,
                    "is_functional": False,
                    "paramValues": params,
                    "varName": var_name,
                    "custom_module_id": custom_module_id,
                    "inputs": [{"id": "in", "name": "Input", "type": "tensor"}],
                    "outputs": [{"id": "out", "name": "Output", "type": "tensor"}],
                    "params": [],
                },
            }

            if block_id in {"custom", "custom_module"}:
                in_ports, out_ports = self._load_custom_module_ports(custom_module_id)
                node_payload["data"]["inputs"] = in_ports
                node_payload["data"]["outputs"] = out_ports
                node_payload["data"]["label"] = label if label != "Custom" else (custom_module_id.split("/")[-1].capitalize() or "Custom")
            else:
                b = get_block_by_id(block_id)
                if b:
                    node_payload["data"]["inputs"] = [p.model_dump() for p in b.definition.inputs]
                    node_payload["data"]["outputs"] = [p.model_dump() for p in b.definition.outputs]
                    node_payload["data"]["params"] = [p.model_dump() for p in b.definition.params]
                    if not node_info.get("label"):
                        node_payload["data"]["label"] = b.definition.name
                    node_payload["data"]["is_functional"] = b.definition.is_functional

                    # Fill missing defaults
                    for p in b.definition.params:
                        if p.name not in node_payload["data"]["paramValues"]:
                            node_payload["data"]["paramValues"][p.name] = p.default

            # Dynamically ensure all handles referenced by edges targeting or sourcing this node exist on the node
            needed_in_handles = []
            for e_item in self.ir_edges:
                _, _, d_a, d_h = self._parse_edge_full(e_item)
                if d_a == alias and d_h and d_h not in needed_in_handles:
                    needed_in_handles.append(d_h)
            
            existing_in_ids = {p["id"] for p in node_payload["data"]["inputs"] if isinstance(p, dict)}
            for h in needed_in_handles:
                if h not in existing_in_ids:
                    node_payload["data"]["inputs"].append({
                        "id": h,
                        "name": h.replace("_", " ").capitalize(),
                        "type": "tensor"
                    })
                    existing_in_ids.add(h)

            needed_out_handles = []
            for e_item in self.ir_edges:
                s_a, s_h, _, _ = self._parse_edge_full(e_item)
                if s_a == alias and s_h and s_h not in needed_out_handles:
                    needed_out_handles.append(s_h)
            
            existing_out_ids = {p["id"] for p in node_payload["data"]["outputs"] if isinstance(p, dict)}
            for h in needed_out_handles:
                if h not in existing_out_ids:
                    node_payload["data"]["outputs"].append({
                        "id": h,
                        "name": h.replace("_", " ").capitalize(),
                        "type": "tensor"
                    })
                    existing_out_ids.add(h)

            self.nodes_data[alias] = node_payload

        # 2. Hydrate Edges
        for edge_idx, edge_item in enumerate(self.ir_edges):
            if isinstance(edge_item, str):
                try:
                    src_part, dst_part = [p.strip() for p in edge_item.split("->")]
                    src_alias, src_handle = [p.strip() for p in src_part.split(".")]
                    dst_alias, dst_handle = [p.strip() for p in dst_part.split(".")]
                except ValueError:
                    raise ValueError(
                        f"Malformed edge string '{edge_item}'. Expected format 'source_alias.port -> target_alias.port'"
                    )
            elif isinstance(edge_item, dict):
                src_alias = edge_item.get("source_alias") or edge_item.get("source", "").split(".")[0]
                src_handle = edge_item.get("source_handle") or (edge_item.get("source", "").split(".")[1] if "." in edge_item.get("source", "") else "out")
                dst_alias = edge_item.get("target_alias") or edge_item.get("target", "").split(".")[0]
                dst_handle = edge_item.get("target_handle") or (edge_item.get("target", "").split(".")[1] if "." in edge_item.get("target", "") else "in")
            else:
                continue

            src_id = self.node_ids.get(src_alias)
            dst_id = self.node_ids.get(dst_alias)

            if not src_id:
                raise ValueError(f"Edge references unknown source node alias: '{src_alias}'")
            if not dst_id:
                raise ValueError(f"Edge references unknown target node alias: '{dst_alias}'")

            src_node = self.nodes_data.get(src_alias, {})
            dst_node = self.nodes_data.get(dst_alias, {})
            src_outputs = src_node.get("data", {}).get("outputs", [])
            dst_inputs = dst_node.get("data", {}).get("inputs", [])

            # Resolve actual source handle
            actual_src_handle = src_handle
            out_ids = [p["id"] for p in src_outputs if isinstance(p, dict) and "id" in p]
            if actual_src_handle not in out_ids:
                if actual_src_handle in ("out", "output") and out_ids:
                    actual_src_handle = out_ids[0]
                elif actual_src_handle.startswith("out_") and actual_src_handle[4:].isdigit():
                    idx = int(actual_src_handle[4:]) - 1
                    if 0 <= idx < len(out_ids):
                        actual_src_handle = out_ids[idx]
                elif out_ids:
                    actual_src_handle = out_ids[0]

            # Resolve actual destination handle
            actual_dst_handle = dst_handle
            in_ids = [p["id"] for p in dst_inputs if isinstance(p, dict) and "id" in p]
            if actual_dst_handle not in in_ids:
                if actual_dst_handle in ("in", "input") and in_ids:
                    actual_dst_handle = in_ids[0]
                elif actual_dst_handle.startswith("in_") and actual_dst_handle[3:].isdigit():
                    idx = int(actual_dst_handle[3:]) - 1
                    if 0 <= idx < len(in_ids):
                        actual_dst_handle = in_ids[idx]
                elif in_ids:
                    actual_dst_handle = in_ids[0]

            edge_id = f"e_{src_id}_{actual_src_handle}_{dst_id}_{actual_dst_handle}_{edge_idx}"
            self.edges_data.append({
                "id": edge_id,
                "source": src_id,
                "sourceHandle": actual_src_handle,
                "target": dst_id,
                "targetHandle": actual_dst_handle,
                "type": "tensor",
            })

        return {
            "name": self.name,
            "variables": self.variables,
            "parameters": self.variables,  # Backwards compatibility
            "nodes": list(self.nodes_data.values()),
            "edges": self.edges_data,
        }

    def validate(self, compiled_graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs shape inference and PyTorch code compilation checks on the compiled graph.
        Returns validation details or raises ShapeError / ValueError with actionable diagnostics.
        """
        graph_dict = compiled_graph or self.compile()

        # Build GraphData object
        nodes_list = []
        for n in graph_dict["nodes"]:
            nd = NodeData(
                block_id=n["data"].get("block_id", ""),
                label=n["data"].get("label", ""),
                is_functional=n["data"].get("is_functional", False),
                paramValues=dict(n["data"].get("paramValues", {})),
                varName=n["data"].get("varName", ""),
                custom_module_id=n["data"].get("custom_module_id", ""),
            )
            nodes_list.append(Node(id=n["id"], data=nd, position=n.get("position")))

        edges_list = [
            Edge(
                id=e["id"],
                source=e["source"],
                sourceHandle=e["sourceHandle"],
                target=e["target"],
                targetHandle=e["targetHandle"],
            )
            for e in graph_dict["edges"]
        ]

        # Convert variables / parameters
        var_models = []
        for v in graph_dict.get("variables", []):
            if isinstance(v, dict):
                var_models.append(
                    ArchVariableModel(
                        id=v.get("id", generate_id("var")),
                        name=v.get("name", "var"),
                        type=v.get("type", "int"),
                        default=v.get("default", 0),
                        description=v.get("description", ""),
                        scope=v.get("scope", "init_param"),
                    )
                )

        main_graph_id = "main"
        graphs: Dict[str, GraphData] = {
            main_graph_id: GraphData(
                name=graph_dict.get("name", "Main"),
                variables=var_models,
                nodes=nodes_list,
                edges=edges_list,
            )
        }

        # Only load dependent graphs from workspace if needed
        needed_deps = {n.data.custom_module_id for n in nodes_list if n.data.custom_module_id}
        while needed_deps:
            curr_dep = needed_deps.pop()
            if curr_dep in graphs or not self.workspace_dir:
                continue

            candidates = [
                os.path.join(self.workspace_dir, f"{curr_dep}.arch"),
                os.path.join(self.workspace_dir, f"{curr_dep}.json"),
                os.path.join(self.workspace_dir, curr_dep),
            ]
            loaded_path = None
            for p in candidates:
                if os.path.isfile(p):
                    loaded_path = p
                    break

            if loaded_path:
                try:
                    with open(loaded_path, "r", encoding="utf-8") as g_file:
                        g_json = json.load(g_file)
                    g_nodes = [
                        Node(
                            id=gn["id"],
                            data=NodeData(
                                block_id=gn["data"].get("block_id", ""),
                                label=gn["data"].get("label", ""),
                                is_functional=gn["data"].get("is_functional", False),
                                paramValues=dict(gn["data"].get("paramValues", {})),
                                varName=gn["data"].get("varName", ""),
                                custom_module_id=gn["data"].get("custom_module_id", ""),
                            ),
                            position=gn.get("position"),
                        )
                        for gn in g_json.get("nodes", [])
                    ]
                    g_edges = [
                        Edge(
                            id=ge["id"],
                            source=ge["source"],
                            sourceHandle=ge["sourceHandle"],
                            target=ge["target"],
                            targetHandle=ge["targetHandle"],
                        )
                        for ge in g_json.get("edges", [])
                    ]
                    g_vars = [
                        ArchVariableModel(
                            id=v.get("id", generate_id("var")),
                            name=v.get("name", "var"),
                            type=v.get("type", "int"),
                            default=v.get("default", 0),
                            description=v.get("description", ""),
                            scope=v.get("scope", "init_param"),
                        )
                        for v in (g_json.get("variables") or g_json.get("parameters") or [])
                        if isinstance(v, dict)
                    ]
                    graphs[curr_dep] = GraphData(
                        name=g_json.get("name", curr_dep),
                        variables=g_vars,
                        nodes=g_nodes,
                        edges=g_edges,
                    )
                    # Add any nested dependencies
                    for gn in g_nodes:
                        if gn.data.custom_module_id and gn.data.custom_module_id not in graphs:
                            needed_deps.add(gn.data.custom_module_id)
                except Exception:
                    pass

        # 1. Topological & Shape validation
        node_shapes, node_params = shape_inference_multi_graph(graphs, main_graph_id)

        # 2. PyTorch Code generation validation
        compiled_files, _, _ = generate_pytorch_code(graphs, main_graph_id)

        # 3. Python Syntax Verification
        main_py_code = compiled_files.get(main_graph_id, "")
        compile(main_py_code, f"<{self.name}>", "exec")

        return {
            "valid": True,
            "node_shapes": node_shapes,
            "node_params": node_params,
            "code": main_py_code,
        }


def decompile_arch_to_ir(
    arch_input: Any,
    output_path: Optional[str] = None,
    ir_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompiles a full ArchIDE .arch (React Flow JSON) into a clean,
    compact Agentic Graph IR (.ir.json).
    """
    if isinstance(arch_input, str):
        with open(arch_input, "r", encoding="utf-8") as f:
            arch_data = json.load(f)
        if not output_path:
            base_name = os.path.basename(arch_input)
            stem = base_name[:-5] if base_name.endswith(".arch") else os.path.splitext(base_name)[0]
            if ir_dir:
                output_path = os.path.join(ir_dir, f"{stem}.ir.json")
            else:
                # If input is in workspace/graphs, place output in sibling workspace/ir
                parent_dir = os.path.dirname(os.path.abspath(arch_input))
                if os.path.basename(parent_dir) == "graphs":
                    ir_sibling = os.path.abspath(os.path.join(parent_dir, "..", "ir"))
                    output_path = os.path.join(ir_sibling, f"{stem}.ir.json")
                else:
                    output_path = arch_input.replace(".arch", ".ir.json")
                    if output_path == arch_input:
                        output_path += ".ir.json"
    elif isinstance(arch_input, dict):
        arch_data = arch_input
    else:
        raise TypeError("arch_input must be a filepath string or dictionary")

    id_to_alias: Dict[str, str] = {}
    alias_counts: Dict[str, int] = {}
    ir_nodes: Dict[str, Dict[str, Any]] = {}

    nodes = arch_data.get("nodes", [])
    for n in nodes:
        node_id = n.get("id", "")
        data = n.get("data", {})
        block_id = data.get("block_id") or "custom"
        var_name = data.get("varName") or ""
        custom_mod_id = data.get("custom_module_id") or ""
        label = data.get("label") or ""

        # Choose a clean base alias
        if block_id in {"input", "gourav"}:
            base_alias = "in"
        elif block_id == "output":
            base_alias = "out"
        elif label and label.lower() not in {"custom", "custom_module", "sequential", "input", "output"} and len(label) < 30:
            import re
            clean_lbl = re.sub(r'[^a-zA-Z0-9_]', '_', label).strip('_').lower()
            if clean_lbl.startswith("module_") and len(clean_lbl) > 7:
                clean_lbl = custom_mod_id.split("/")[-1].lower() if custom_mod_id else clean_lbl
            base_alias = clean_lbl or block_id
        elif custom_mod_id:
            base_alias = custom_mod_id.split("/")[-1].lower()
        elif var_name and var_name.lower() != "x":
            base_alias = var_name
        else:
            base_alias = block_id

        # Disambiguate duplicate aliases
        alias_counts[base_alias] = alias_counts.get(base_alias, 0) + 1
        if alias_counts[base_alias] == 1:
            alias = base_alias
        else:
            alias = f"{base_alias}_{alias_counts[base_alias]}"

        id_to_alias[node_id] = alias

        node_entry: Dict[str, Any] = {"block": block_id}
        if custom_mod_id:
            node_entry["custom_module_id"] = custom_mod_id
        if var_name:
            node_entry["var_name"] = var_name

        param_values = dict(data.get("paramValues", {}))
        cleaned_params = {k: v for k, v in param_values.items() if not k.startswith("_") and v is not None}
        if cleaned_params and block_id != "output":
            node_entry["params"] = cleaned_params

        ir_nodes[alias] = node_entry

    import re
    node_by_id = {n.get("id"): n for n in nodes}
    ir_edges: List[str] = []
    edges = arch_data.get("edges", [])
    for e in edges:
        src_id = e.get("source", "")
        src_handle = e.get("sourceHandle") or "out"
        dst_id = e.get("target", "")
        dst_handle = e.get("targetHandle") or "in"

        src_alias = id_to_alias.get(src_id, src_id)
        dst_alias = id_to_alias.get(dst_id, dst_id)

        # Normalize source handle
        if src_handle in ("out", "out_1", "output", ""):
            clean_src_handle = "out"
        elif re.match(r"^out_\d+$", src_handle):
            clean_src_handle = src_handle
        elif src_handle in ("attn", "scores", "weights", "features"):
            clean_src_handle = src_handle
        else:
            src_node = node_by_id.get(src_id, {})
            src_outputs = src_node.get("data", {}).get("outputs", [])
            out_ids = [p.get("id") for p in src_outputs if isinstance(p, dict)]
            if src_handle in out_ids:
                idx = out_ids.index(src_handle)
                clean_src_handle = "out" if idx == 0 else f"out_{idx + 1}"
            else:
                clean_src_handle = "out"

        # Normalize destination handle
        if dst_handle in ("in", "in_1", "input", ""):
            clean_dst_handle = "in"
        elif dst_handle in ("in_a", "in_b"):
            clean_dst_handle = dst_handle
        elif re.match(r"^in_\d+$", dst_handle):
            clean_dst_handle = dst_handle
        elif dst_handle in ("q", "k", "v", "residual", "context", "features", "weight", "bias"):
            clean_dst_handle = dst_handle
        else:
            dst_node = node_by_id.get(dst_id, {})
            dst_inputs = dst_node.get("data", {}).get("inputs", [])
            in_ids = [p.get("id") for p in dst_inputs if isinstance(p, dict)]
            if dst_handle in in_ids:
                idx = in_ids.index(dst_handle)
                clean_dst_handle = "in" if idx == 0 else f"in_{idx + 1}"
            else:
                clean_dst_handle = dst_handle or "in"

        ir_edges.append(f"{src_alias}.{clean_src_handle} -> {dst_alias}.{clean_dst_handle}")

    variables = arch_data.get("variables") or arch_data.get("parameters") or []
    ir_payload: Dict[str, Any] = {
        "name": arch_data.get("name", "Model"),
        "variables": variables,
        "nodes": ir_nodes,
        "edges": ir_edges,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ir_payload, f, indent=2)

    return ir_payload


def compile_ir(
    ir_input: Any,
    output_path: Optional[str] = None,
    validate: bool = True,
    workspace_dir: Optional[str] = None,
    arch_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to compile and optionally validate an IR payload or file."""
    if isinstance(ir_input, str):
        with open(ir_input, "r", encoding="utf-8") as f:
            ir_payload = json.load(f)
        if not output_path:
            base_name = os.path.basename(ir_input)
            stem = base_name.replace(".ir.json", "").replace(".json", "")
            if arch_dir:
                output_path = os.path.join(arch_dir, f"{stem}.arch")
            else:
                parent_dir = os.path.dirname(os.path.abspath(ir_input))
                if os.path.basename(parent_dir) == "ir":
                    graphs_sibling = os.path.abspath(os.path.join(parent_dir, "..", "graphs"))
                    output_path = os.path.join(graphs_sibling, f"{stem}.arch")
                else:
                    output_path = ir_input.replace(".ir.json", ".arch").replace(".json", ".arch")
                    if output_path == ir_input:
                        output_path += ".arch"
    elif isinstance(ir_input, dict):
        ir_payload = ir_input
    else:
        raise TypeError("ir_input must be a filepath string or dictionary")

    compiler = AgentGraphCompiler(ir_payload, workspace_dir=workspace_dir)
    compiled_graph = compiler.compile()

    if validate:
        compiler.validate(compiled_graph)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compiled_graph, f, indent=2)

    return compiled_graph


from python_decompiler import decompile_python_to_ir, PyTorchASTDecompiler


def main():
    parser = argparse.ArgumentParser(description="ArchIDE Agent Graph Compiler & Decompiler")
    parser.add_argument("input", nargs="?", help="Input file path (.ir.json, .arch, or .py)")
    parser.add_argument("-o", "--output", help="Output file path (optional)")
    parser.add_argument("--to-ir", action="store_true", help="Decompile .arch file into .ir.json")
    parser.add_argument("--to-arch", action="store_true", help="Compile .ir.json file into .arch")
    parser.add_argument("--from-python", action="store_true", help="Decompile .py file into .ir.json")
    parser.add_argument("--all-to-ir", action="store_true", help="Decompile all .arch files in workspace/graphs into workspace/ir/*.ir.json")
    parser.add_argument("--all-to-arch", action="store_true", help="Compile all .ir.json files in workspace/ir into workspace/graphs/*.arch with validation")
    parser.add_argument("--all-from-python", action="store_true", help="Decompile all .py files in workspace/python into workspace/ir/*.ir.json")
    parser.add_argument("--all-to-python", action="store_true", help="Compile all .arch files in workspace/graphs into workspace/python/*.py")
    parser.add_argument("--no-validate", action="store_true", help="Skip shape & compiler validation pass")
    parser.add_argument("--workspace", help="Workspace graphs directory for submodule resolution", default=None)
    parser.add_argument("--arch-dir", help="Directory containing .arch files (default: workspace/graphs)", default=None)
    parser.add_argument("--ir-dir", help="Directory containing .ir.json files (default: workspace/ir)", default=None)
    parser.add_argument("--python-dir", help="Directory containing .py files (default: workspace/python)", default=None)

    args = parser.parse_args()

    default_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "../workspace"))
    arch_dir = os.path.abspath(args.arch_dir or os.path.join(default_workspace, "graphs"))
    ir_dir = os.path.abspath(args.ir_dir or os.path.join(default_workspace, "ir"))
    python_dir = os.path.abspath(args.python_dir or os.path.join(default_workspace, "python"))
    workspace_dir = os.path.abspath(args.workspace or arch_dir)

    if args.all_to_python:
        if not os.path.isdir(arch_dir):
            print(f"Arch directory not found: {arch_dir}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(python_dir, exist_ok=True)
        graphs = {}
        file_paths = {}
        for root, _, files in os.walk(arch_dir):
            for f in sorted(files):
                if f.endswith(".arch"):
                    full_path = os.path.join(root, f)
                    rel = os.path.relpath(full_path, arch_dir)
                    key = rel[:-5]
                    with open(full_path, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                    nodes = [Node(id=n["id"], data=NodeData(**n["data"]), position=n.get("position")) for n in d["nodes"]]
                    edges = [Edge(**e) for e in d["edges"]]
                    vars = [ArchVariableModel(**v) for v in (d.get("variables") or d.get("parameters") or []) if isinstance(v, dict)]
                    graphs[key] = GraphData(name=d.get("name", key), nodes=nodes, edges=edges, variables=vars)
                    file_paths[key] = key

        compiled_files, _, _ = generate_pytorch_code(graphs, "main", file_paths=file_paths)
        for k, code in compiled_files.items():
            out_file = os.path.join(python_dir, f"{k}.py")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as fp:
                fp.write(code)
            print(f"Generated PyTorch: {os.path.relpath(out_file)}")
        print(f"Successfully compiled {len(compiled_files)} models to {os.path.relpath(python_dir)}")
        return

    if args.all_from_python:
        if not os.path.isdir(python_dir):
            print(f"Python directory not found: {python_dir}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(ir_dir, exist_ok=True)
        count = 0
        for root, _, files in os.walk(python_dir):
            for f in sorted(files):
                if f.endswith(".py") and not f.startswith("__"):
                    py_file = os.path.join(root, f)
                    rel_subpath = os.path.relpath(py_file, python_dir)
                    stem = rel_subpath[:-3]
                    out_ir = os.path.join(ir_dir, f"{stem}.ir.json")
                    res = decompile_python_to_ir(py_file, output_path=out_ir, workspace_dir=workspace_dir)
                    print(f"Decompiled PyTorch: {os.path.relpath(py_file)} -> {os.path.relpath(out_ir)} ({len(res['nodes'])} nodes, {len(res['edges'])} edges)")
                    count += 1
        print(f"Successfully decompiled {count} .py files to {os.path.relpath(ir_dir)}")
        return

    if args.all_to_ir:
        if not os.path.isdir(arch_dir):
            print(f"Arch directory not found: {arch_dir}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(ir_dir, exist_ok=True)
        count = 0
        for root, _, files in os.walk(arch_dir):
            for f in sorted(files):
                if f.endswith(".arch"):
                    arch_file = os.path.join(root, f)
                    rel_subpath = os.path.relpath(arch_file, arch_dir)
                    stem = rel_subpath[:-5]
                    out_ir = os.path.join(ir_dir, f"{stem}.ir.json")
                    res = decompile_arch_to_ir(arch_file, output_path=out_ir)
                    print(f"Exported IR: {os.path.relpath(arch_file)} -> {os.path.relpath(out_ir)} ({len(res['nodes'])} nodes, {len(res['edges'])} edges)")
                    count += 1
        print(f"Successfully decompiled {count} .arch files to {os.path.relpath(ir_dir)}")
        return

    if args.all_to_arch:
        if not os.path.isdir(ir_dir):
            print(f"IR directory not found: {ir_dir}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(arch_dir, exist_ok=True)
        count = 0
        errors = []
        for root, _, files in os.walk(ir_dir):
            for f in sorted(files):
                if f.endswith(".ir.json") or f.endswith(".json"):
                    ir_file = os.path.join(root, f)
                    rel_subpath = os.path.relpath(ir_file, ir_dir)
                    stem = rel_subpath.replace(".ir.json", "").replace(".json", "")
                    out_arch = os.path.join(arch_dir, f"{stem}.arch")
                    try:
                        res = compile_ir(
                            ir_file,
                            output_path=out_arch,
                            validate=not args.no_validate,
                            workspace_dir=workspace_dir,
                            arch_dir=arch_dir,
                        )
                        print(f"Compiled & Validated: {os.path.relpath(ir_file)} -> {os.path.relpath(out_arch)} ({len(res['nodes'])} nodes, {len(res['edges'])} edges)")
                        count += 1
                    except ShapeError as se:
                        err_msg = f"SHAPE ERROR in {f}: {se} (Node: {se.node_label} [{se.node_id}])"
                        print(err_msg, file=sys.stderr)
                        errors.append(err_msg)
                    except Exception as e:
                        err_msg = f"ERROR in {f}: {e}"
                        print(err_msg, file=sys.stderr)
                        errors.append(err_msg)

        if errors:
            print(f"\nCompilation completed with {len(errors)} error(s) out of {count + len(errors)} files.", file=sys.stderr)
            sys.exit(1)
        print(f"Successfully compiled and validated {count} IR files to {os.path.relpath(arch_dir)}")

        # Synchronize generated PyTorch code to python_dir
        if count > 0:
            os.makedirs(python_dir, exist_ok=True)
            graphs = {}
            file_paths = {}
            for root, _, files in os.walk(arch_dir):
                for f in sorted(files):
                    if f.endswith(".arch"):
                        full_path = os.path.join(root, f)
                        rel = os.path.relpath(full_path, arch_dir)
                        key = rel[:-5]
                        with open(full_path, "r", encoding="utf-8") as fp:
                            d = json.load(fp)
                        nodes = [Node(id=n["id"], data=NodeData(**n["data"]), position=n.get("position")) for n in d["nodes"]]
                        edges = [Edge(**e) for e in d["edges"]]
                        vars = [ArchVariableModel(**v) for v in (d.get("variables") or d.get("parameters") or []) if isinstance(v, dict)]
                        graphs[key] = GraphData(name=d.get("name", key), nodes=nodes, edges=edges, variables=vars)
                        file_paths[key] = key

            try:
                compiled_files, _, _ = generate_pytorch_code(graphs, "main", file_paths=file_paths)
                for k, code in compiled_files.items():
                    out_file = os.path.join(python_dir, f"{k}.py")
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    with open(out_file, "w", encoding="utf-8") as fp:
                        fp.write(code)
            except Exception:
                pass

        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    is_python_input = (args.input.endswith(".py") or args.from_python)
    is_arch_input = (args.input.endswith(".arch") or args.to_ir) and not args.to_arch and not is_python_input

    try:
        if is_python_input:
            result = decompile_python_to_ir(args.input, output_path=args.output, ir_dir=ir_dir, workspace_dir=workspace_dir)
            out_target = args.output or os.path.join(ir_dir, f"{os.path.basename(args.input).replace('.py', '')}.ir.json")
            print(f"SUCCESS: Decompiled Python {args.input} -> {out_target} ({len(result['nodes'])} nodes, {len(result['edges'])} edges)")
        elif is_arch_input:
            result = decompile_arch_to_ir(args.input, output_path=args.output, ir_dir=ir_dir)
            out_target = args.output or os.path.join(ir_dir, f"{os.path.basename(args.input).replace('.arch', '')}.ir.json")
            print(f"SUCCESS: Decompiled {args.input} -> {out_target} ({len(result['nodes'])} nodes, {len(result['edges'])} edges)")
        else:
            result = compile_ir(
                args.input,
                output_path=args.output,
                validate=not args.no_validate,
                workspace_dir=workspace_dir,
                arch_dir=arch_dir,
            )
            out_target = args.output or os.path.join(arch_dir, f"{os.path.basename(args.input).replace('.ir.json', '').replace('.json', '')}.arch")
            print(f"SUCCESS: Compiled and validated {args.input} -> {out_target} ({len(result['nodes'])} nodes, {len(result['edges'])} edges)")
    except ShapeError as se:
        print(f"SHAPE ERROR: {se} (Node: {se.node_label} [{se.node_id}])", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
