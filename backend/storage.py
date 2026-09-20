import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace" / "graphs"

def resolve_dir(dir_path: str) -> Path:
    p = Path(dir_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()

class GraphStorage:
    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir if workspace_dir is not None else WORKSPACE_DIR
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

    def mirror_vfs_state(self, dest_dir: str, files: Dict[str, Any]) -> str:
        """
        Mirrors the browser VFS state to dest_dir with exact file paths and names.
        `files` maps relative paths (e.g. "graphs/main.arch", "python/main.py", "archide.toml")
        to their contents (dict/list for JSON/.arch, string for code/toml).

        Prunes any extraneous files in dest_dir that are not in `files`,
        guaranteeing an exact 1:1 mirror without redundant or zombie files.
        """
        dest = resolve_dir(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        expected_rel_paths = set(files.keys())

        # 1. Prune redundant files: delete any existing file in dest not present in files
        for root, dirs, filenames in os.walk(dest, topdown=False):
            root_path = Path(root)
            # Skip hidden/git folders
            try:
                if any(part.startswith('.') for part in root_path.relative_to(dest).parts):
                    continue
            except ValueError:
                continue

            for fname in filenames:
                if fname.startswith('.') or fname == 'archide_builder.py':
                    continue
                fpath = root_path / fname
                rel_path = str(fpath.relative_to(dest)).replace("\\", "/")
                if rel_path not in expected_rel_paths:
                    try:
                        fpath.unlink()
                    except Exception as e:
                        print(f"Failed to remove redundant file {fpath}: {e}")

            # Remove empty directories (except dest root)
            if root_path != dest:
                try:
                    if not any(root_path.iterdir()):
                        root_path.rmdir()
                except Exception:
                    pass

        # 2. Write out exact files as defined in browser VFS
        for rel_path, content in files.items():
            fpath = dest / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, (dict, list)):
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2)
            else:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(str(content) if content is not None else "")

        # 3. Update workspace_dir so subsequent node/edge auto-saves target the mirrored graphs folder
        graphs_dir = dest / "graphs"
        if graphs_dir.exists():
            self.workspace_dir = graphs_dir
        else:
            self.workspace_dir = dest

        return str(dest)

    def load_vfs_state(self, src_dir: str) -> Dict[str, Any]:
        """
        Scans src_dir recursively and returns:
          { "files": { [rel_path: string]: content_or_string } }
        """
        src = resolve_dir(src_dir)
        if not src.exists():
            raise FileNotFoundError(f"Directory does not exist: {src_dir}")

        files: Dict[str, Any] = {}
        for root, dirs, filenames in os.walk(src):
            root_path = Path(root)
            try:
                if any(part.startswith('.') for part in root_path.relative_to(src).parts):
                    continue
            except ValueError:
                continue

            for fname in filenames:
                if fname.startswith('.') or fname == 'archide_builder.py':
                    continue
                fpath = root_path / fname
                rel_path = str(fpath.relative_to(src)).replace("\\", "/")
                try:
                    if fname.endswith(".arch") or fname.endswith(".json"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            files[rel_path] = json.load(f)
                    else:
                        with open(fpath, "r", encoding="utf-8") as f:
                            files[rel_path] = f.read()
                except Exception:
                    pass

        # Update active workspace_dir if graphs directory exists
        if (src / "graphs").exists():
            self.workspace_dir = src / "graphs"

        return {"files": files}

    def save_state_bundle(
        self,
        dest_dir: str,
        graphs: Optional[Dict[str, Any]] = None,
        python: Optional[Dict[str, str]] = None,
        python_dir: Optional[str] = None
    ) -> str:
        dest = resolve_dir(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        if graphs is None:
            graphs = self.get_all_graphs()

        graphs_dest = dest / "graphs"
        graphs_dest.mkdir(parents=True, exist_ok=True)
        for file_id, content in graphs.items():
            clean_id = file_id[:-5] if file_id.endswith(".arch") else file_id
            fpath = graphs_dest / f"{clean_id}.arch"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

        if python is None:
            python = {}
            py_root = resolve_dir(python_dir) if python_dir else self.workspace_dir.parent / "python"
            if py_root.exists():
                for filepath in py_root.rglob("*.py"):
                    rel = str(filepath.relative_to(py_root)).replace("\\", "/")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            python[rel] = f.read()
                    except Exception:
                        pass

        python_dest = dest / "python"
        python_dest.mkdir(parents=True, exist_ok=True)
        for rel_path, code in python.items():
            py_fpath = python_dest / rel_path
            py_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(py_fpath, "w", encoding="utf-8") as f:
                f.write(code)

        bundle = {"graphs": graphs, "python": python}
        out_path = dest / "state.archstate"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

        self.workspace_dir = graphs_dest
        return str(out_path)

    def load_state_bundle(self, src_dir: str) -> Dict[str, Any]:
        src = resolve_dir(src_dir)
        if not src.exists():
            raise FileNotFoundError(f"Directory does not exist: {src_dir}")

        graphs: Dict[str, Any] = {}
        python: Dict[str, str] = {}

        bundle_path = src / "state.archstate"
        if bundle_path.exists():
            with open(bundle_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
                graphs.update(bundle.get("graphs", {}))
                python.update(bundle.get("python", {}))

        graphs_dir = src / "graphs" if (src / "graphs").exists() else src
        for filepath in graphs_dir.rglob("*.arch"):
            rel_path = filepath.relative_to(graphs_dir)
            file_id = str(rel_path.with_suffix("")).replace("\\", "/")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    graphs[file_id] = json.load(f)
            except Exception:
                pass

        py_dir = src / "python" if (src / "python").exists() else src
        for filepath in py_dir.rglob("*.py"):
            rel = str(filepath.relative_to(py_dir)).replace("\\", "/")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    python[rel] = f.read()
            except Exception:
                pass

        if not graphs and not python and not bundle_path.exists():
            raise FileNotFoundError(f"No state.archstate, .arch, or .py files found in: {src_dir}")

        if (src / "graphs").exists():
            self.workspace_dir = src / "graphs"

        return {"graphs": graphs, "python": python}

# Singleton instance for the application
storage = GraphStorage()
