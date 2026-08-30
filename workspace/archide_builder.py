import json
import os
import uuid
from typing import List, Dict, Any, Optional

def generate_id():
    return uuid.uuid4().hex[:8]

class ArchIDEGraph:
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        self.edges = []
        
    def add_node(self, block_id: str, x: float, y: float, params: Dict[str, Any] = None, custom_module_id: str = "") -> str:
        """Adds a node and returns its generated ID. Automatically populates schemas if run in backend environment."""
        node_id = f"{block_id}_{generate_id()}"
        
        node = {
            "id": node_id,
            "type": "custom",
            "position": {"x": x, "y": y},
            "data": {
                "block_id": block_id,
                "label": block_id.capitalize(),
                "is_functional": False,
                "paramValues": params or {},
                "custom_module_id": custom_module_id,
                "inputs": [{"id": "in", "name": "Input", "type": "tensor"}],
                "outputs": [{"id": "out", "name": "Output", "type": "tensor"}],
                "params": []
            }
        }
        
        try:
            import sys
            backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend'))
            if backend_path not in sys.path: sys.path.append(backend_path)
            from blocks import get_block_by_id
            b = get_block_by_id(block_id)
            if b:
                node["data"]["inputs"] = [p.model_dump() for p in b.definition.inputs]
                node["data"]["outputs"] = [p.model_dump() for p in b.definition.outputs]
                node["data"]["params"] = [p.model_dump() for p in b.definition.params]
                node["data"]["label"] = b.definition.name
                node["data"]["is_functional"] = b.definition.is_functional
                for p in b.definition.params:
                    if p.name not in node["data"]["paramValues"]:
                        node["data"]["paramValues"][p.name] = p.default
        except Exception:
            pass

        self.nodes.append(node)
        return node_id
        
    def add_edge(self, source_id: str, source_handle: str, target_id: str, target_handle: str):
        self.edges.append({
            "id": f"e_{source_id}_{target_id}_{target_handle}",
            "source": source_id,
            "sourceHandle": source_handle,
            "target": target_id,
            "targetHandle": target_handle,
            "type": "tensor"
        })
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({"name": self.name, "nodes": self.nodes, "edges": self.edges, "parameters": []}, f, indent=2)
        print(f"Saved graph to {filepath}")
