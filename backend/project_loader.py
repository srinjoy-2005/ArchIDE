"""
backend/project_loader.py

Headless project loader and compiler helper for ArchIDE.
Loads multi-graph models and custom modules directly from project folders or JSON files,
enabling automated testing, CLI workflows, and headless model compilation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
from models import (
    CompileRequest,
    CheckRequest,
    GraphData,
    Node,
    Edge,
    NodeData,
    ParamDef,
    ArchIDEProject,
    ProjectFileData,
    FolderData,
)
from compiler import generate_pytorch_code, shape_inference_multi_graph


def _parse_graph_from_dict(data: Dict[str, Any], default_name: str = "Model") -> GraphData:
    """Parse a dictionary into a validated GraphData object."""
    name = data.get("name", default_name)
    
    # Parse parameters
    raw_params = data.get("parameters", [])
    params = []
    for p in raw_params:
        if isinstance(p, dict):
            params.append(ParamDef(**p))
        elif isinstance(p, ParamDef):
            params.append(p)
            
    # Parse nodes
    raw_nodes = data.get("nodes", [])
    nodes = []
    for n in raw_nodes:
        if isinstance(n, dict):
            d = n.get("data", {})
            node_data = NodeData(
                block_id=d.get("block_id", ""),
                label=d.get("label", ""),
                is_functional=d.get("is_functional", False),
                paramValues=d.get("paramValues", {}),
                varName=d.get("varName", ""),
                custom_module_id=d.get("custom_module_id", ""),
            )
            nodes.append(Node(id=n.get("id", ""), data=node_data, position=n.get("position")))
        elif isinstance(n, Node):
            nodes.append(n)
            
    # Parse edges
    raw_edges = data.get("edges", [])
    edges = []
    for e in raw_edges:
        if isinstance(e, dict):
            edges.append(Edge(**e))
        elif isinstance(e, Edge):
            edges.append(e)
            
    return GraphData(name=name, parameters=params, nodes=nodes, edges=edges)


def load_project(project_path: Union[str, Path]) -> CompileRequest:
    """
    Loads a multi-module ArchIDE project from a directory, manifest file, or JSON file.
    
    Returns:
        CompileRequest containing all parsed graphs and the main graph ID.
    """
    path = Path(project_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project path does not exist: {path}")

    graphs: Dict[str, GraphData] = {}
    main_graph_id: Optional[str] = None

    if path.is_dir():
        manifest_path = path / "archide.project.json"
        if manifest_path.exists():
            return _load_from_manifest(manifest_path)
        
        # Discover all JSON files recursively
        json_files = list(path.glob("**/*.json"))
        if not json_files:
            raise ValueError(f"No JSON graph files found in directory: {path}")
            
        for jf in json_files:
            file_id = jf.stem
            with open(jf, "r", encoding="utf-8") as f:
                content = json.load(f)
            graphs[file_id] = _parse_graph_from_dict(content, default_name=file_id.capitalize())
            if file_id.lower() in {"main", "main_graph", "model"} and main_graph_id is None:
                main_graph_id = file_id

        if main_graph_id is None and graphs:
            main_graph_id = list(graphs.keys())[0]

    elif path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
            
        # Check if it's an ArchIDEProject manifest structure
        if "files" in content or "entry_point" in content:
            return _load_from_manifest_dict(content, base_dir=path.parent)
            
        # Otherwise, treat as a single graph file
        file_id = path.stem
        graphs[file_id] = _parse_graph_from_dict(content, default_name=file_id.capitalize())
        main_graph_id = file_id

    if not graphs or not main_graph_id:
        raise ValueError(f"Failed to load any valid graphs from: {path}")

    return CompileRequest(main_graph_id=main_graph_id, graphs=graphs)


def _load_from_manifest(manifest_path: Path) -> CompileRequest:
    with open(manifest_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return _load_from_manifest_dict(content, base_dir=manifest_path.parent)


def _load_from_manifest_dict(manifest: Dict[str, Any], base_dir: Path) -> CompileRequest:
    graphs: Dict[str, GraphData] = {}
    entry_point = manifest.get("entry_point", manifest.get("entryPointId", "main"))
    
    files = manifest.get("files", [])
    for item in files:
        file_id = item.get("id", "")
        # If nodes/edges are inlined directly in the manifest
        if "nodes" in item and item["nodes"]:
            graphs[file_id] = _parse_graph_from_dict(item, default_name=item.get("name", file_id))
        elif "path" in item and item["path"]:
            file_path = base_dir / item["path"]
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = json.load(f)
                graphs[file_id] = _parse_graph_from_dict(file_content, default_name=item.get("name", file_id))
            else:
                # Create empty graph placeholder if file missing
                graphs[file_id] = GraphData(name=item.get("name", file_id), nodes=[], edges=[])
        else:
            graphs[file_id] = _parse_graph_from_dict(item, default_name=item.get("name", file_id))

    if entry_point not in graphs and graphs:
        # Fallback to first available graph
        entry_point = list(graphs.keys())[0]

    return CompileRequest(main_graph_id=entry_point, graphs=graphs)


def compile_project(project_path: Union[str, Path]) -> str:
    """Loads a project from path and compiles it to PyTorch source code."""
    req = load_project(project_path)
    return generate_pytorch_code(req.graphs, req.main_graph_id)


def check_project(project_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a project from path and runs static shape inference."""
    req = load_project(project_path)
    shapes, params = shape_inference_multi_graph(req.graphs, req.main_graph_id)
    return {"node_shapes": shapes, "node_params": params}
