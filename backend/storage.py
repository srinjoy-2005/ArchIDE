import os
import json
from pathlib import Path
from typing import Dict, Any

WORKSPACE_DIR = Path(__file__).parent.parent / "workspace" / "graphs"

class GraphStorage:
    def __init__(self, workspace_dir: Path = WORKSPACE_DIR):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        
    def get_all_graphs(self) -> Dict[str, Any]:
        """Recursively finds all .arch files and returns dict mapping relative path -> content"""
        files = {}
        if os.path.exists(self.workspace_dir):
            for filepath in self.workspace_dir.rglob("*.arch"):
                rel_path = filepath.relative_to(self.workspace_dir)
                # e.g. "conv/res_block.arch" -> "conv/res_block"
                file_id = str(rel_path.with_suffix("")).replace("\\", "/")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        files[file_id] = json.load(f)
                except Exception:
                    pass
        return files
        
    def save_graph(self, file_id: str, content: Dict[str, Any]) -> bool:
        """Saves a graph file. Returns True if successful."""
        try:
            filepath = self.workspace_dir / f"{file_id}.arch"
            os.makedirs(filepath.parent, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save {file_id}: {e}")
            return False
            
    def get_graph_mtimes(self) -> Dict[str, float]:
        """Returns a dict of relative paths to their last modified timestamp."""
        mtimes = {}
        if os.path.exists(self.workspace_dir):
            for filepath in self.workspace_dir.rglob("*.arch"):
                rel_path = str(filepath.relative_to(self.workspace_dir)).replace("\\", "/")
                try:
                    mtimes[rel_path] = os.path.getmtime(filepath)
                except FileNotFoundError:
                    pass
        return mtimes
        
    def read_graph(self, rel_path: str) -> Dict[str, Any]:
        """Reads a graph by its exact relative file path (including .arch)."""
        filepath = self.workspace_dir / rel_path
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

# Singleton instance for the application
storage = GraphStorage()
