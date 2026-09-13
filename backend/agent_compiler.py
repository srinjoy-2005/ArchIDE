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

    def _calculate_layout(self) -> Dict[str, Tuple[float, float]]:
        """Assigns non-overlapping (x, y) coordinates based on topological layers."""
        layers = self._assign_layers()
        layer_groups: Dict[int, List[str]] = {}
        for alias, layer_idx in layers.items():
            layer_groups.setdefault(layer_idx, []).append(alias)

        coords: Dict[str, Tuple[float, float]] = {}
        X_SPACING = 300
        Y_SPACING = 150

        for layer_idx, aliases in layer_groups.items():
            x = layer_idx * X_SPACING
            total_height = (len(aliases) - 1) * Y_SPACING
            start_y = -(total_height / 2)

            for i, alias in enumerate(aliases):
                y = start_y + (i * Y_SPACING)
                coords[alias] = (round(x + 100, 1), round(y + 250, 1))

        return coords

    def _parse_edge_aliases(self, edge_item: Any) -> Tuple[Optional[str], Optional[str]]:
        """Extracts source and target aliases from string 'src.port -> dst.port' or dict."""
        if isinstance(edge_item, str):
            try:
                src_part, dst_part = [p.strip() for p in edge_item.split("->")]
                return src_part.split(".")[0].strip(), dst_part.split(".")[0].strip()
            except Exception:
                return None, None
        elif isinstance(edge_item, dict):
            src = edge_item.get("source") or edge_item.get("from", "")
            dst = edge_item.get("target") or edge_item.get("to", "")
            return src.split(".")[0].strip(), dst.split(".")[0].strip()
        return None, None

    def _load_custom_module_ports(self, custom_module_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Attempts to load inputs/outputs port definitions from a referenced .arch file."""
        if not self.workspace_dir or not custom_module_id:
            return [{"id": "in", "name": "Input", "type": "tensor"}], [{"id": "out", "name": "Output", "type": "tensor"}]

        # Check path variations
        candidates = [
            os.path.join(self.workspace_dir, f"{custom_module_id}.arch"),
            os.path.join(self.workspace_dir, f"{custom_module_id}.json"),
            os.path.join(self.workspace_dir, custom_module_id),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    inputs, outputs = [], []
                    for n in data.get("nodes", []):
                        b_id = n.get("data", {}).get("block_id")
                        if b_id in {"input", "gourav"}:
                            for out_p in n.get("data", {}).get("outputs", []):
                                inputs.append({"id": f"in_{len(inputs)+1}", "name": out_p.get("name", "Input"), "type": "tensor"})
                        elif b_id == "output":
                            for in_p in n.get("data", {}).get("inputs", []):
                                outputs.append({"id": f"out_{len(outputs)+1}", "name": in_p.get("name", "Output"), "type": "tensor"})
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

            edge_id = f"e_{src_id}_{src_handle}_{dst_id}_{dst_handle}_{edge_idx}"
            self.edges_data.append({
                "id": edge_id,
                "source": src_id,
                "sourceHandle": src_handle,
                "target": dst_id,
                "targetHandle": dst_handle,
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
                    graphs[curr_dep] = GraphData(
                        name=g_json.get("name", curr_dep),
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


def compile_ir(
    ir_input: Any,
    output_path: Optional[str] = None,
    validate: bool = True,
    workspace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to compile and optionally validate an IR payload or file."""
    if isinstance(ir_input, str):
        with open(ir_input, "r", encoding="utf-8") as f:
            ir_payload = json.load(f)
        if not output_path:
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


def main():
    parser = argparse.ArgumentParser(description="ArchIDE Agent Graph Compiler & Validator")
    parser.add_argument("input", help="Path to input .ir.json file")
    parser.add_argument("-o", "--output", help="Path to output .arch file", default=None)
    parser.add_argument("--no-validate", action="store_true", help="Skip shape & compiler validation pass")
    parser.add_argument("--workspace", help="Workspace graphs directory for submodule resolution", default=None)

    args = parser.parse_args()

    try:
        out_path = args.output
        if not out_path:
            out_path = args.input.replace(".ir.json", ".arch").replace(".json", ".arch")
            if out_path == args.input:
                out_path += ".arch"

        result = compile_ir(
            args.input,
            output_path=out_path,
            validate=not args.no_validate,
            workspace_dir=args.workspace,
        )
        print(f"SUCCESS: Compiled and validated {args.input} -> {out_path} ({len(result['nodes'])} nodes, {len(result['edges'])} edges)")
    except ShapeError as se:
        print(f"SHAPE ERROR: {se.message} (Node: {se.node_label} [{se.node_id}])", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"COMPILATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
