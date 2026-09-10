from fastapi import HTTPException, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from models import BlockDef, CompileRequest, CheckRequest
from blocks import get_all_block_defs
from compiler import topological_sort, generate_pytorch_code, shape_inference_pass, ShapeError
from storage import storage
import json
import os

def print_payloads(req_model, resp_dict):
    print("\n\033[96m--- INCOMING PAYLOAD ---")
    print(json.dumps(req_model.model_dump(), indent=2))
    print("------------------------\033[0m\n")
    print("\n\033[92m--- OUTGOING PAYLOAD ---")
    print(json.dumps(resp_dict, indent=2))
    print("------------------------\033[0m\n")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from graphVFS import router as vfs_router
app.include_router(vfs_router)

@app.get("/api/blocks", response_model=List[BlockDef])
def get_blocks():
    return get_all_block_defs()

@app.get("/api/blocks/{block_id}/docs")
def get_block_docs(block_id: str):
    from blocks import get_block_by_id
    block = get_block_by_id(block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    return block.docs()

@app.post("/api/compile")
def compile_graph(request: CompileRequest):
    try:
        files, node_shapes, node_params = generate_pytorch_code(
            request.graphs, request.main_graph_id, request.file_paths
        )
        
        # Dump files to workspace/python
        python_dir = os.path.join(os.path.dirname(__file__), '../workspace/python')
        
        # Clean stale files from previous compilations
        if os.path.exists(python_dir):
            import shutil
            shutil.rmtree(python_dir)
        os.makedirs(python_dir, exist_ok=True)
        
        for path_key, code_content in files.items():
            # ensure directory for nested paths
            out_file = os.path.join(python_dir, f"{path_key}.py")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(code_content)
                
        # Convert tuples to lists for JSON serialisation
        serialisable_shapes = {
            node_id: {port: list(shape) for port, shape in ports.items()}
            for node_id, ports in node_shapes.items()
        }
                
        resp = {"files": files, "node_shapes": serialisable_shapes, "node_params": node_params}
        print_payloads(request, resp)
        return resp
    except ShapeError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ShapeMismatch",
                "message": str(e),
                "node_id": e.node_id,
                "node_label": e.node_label,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/check")
def check_shapes(request: CheckRequest):
    """
    Runs the static shape inference pass without generating code.
    Returns per-node output shapes and auto-inferred parameters, or a structured error on the first mismatch.
    """
    try:
        from compiler import shape_inference_multi_graph
        node_shapes, node_params = shape_inference_multi_graph(request.graphs, request.main_graph_id)
        # Convert tuples to lists for JSON serialisation
        serialisable = {
            node_id: {port: list(shape) for port, shape in ports.items()}
            for node_id, ports in node_shapes.items()
        }
        resp = {"ok": True, "node_shapes": serialisable, "node_params": node_params}
        print_payloads(request, resp)
        return resp
    except ShapeError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ShapeMismatch",
                "message": str(e),
                "node_id": e.node_id,
                "node_label": e.node_label,
                "edge_ids": e.edge_ids,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)


# ─── State Save / Load ───────────────────────────────────────────────────────

class StateSaveRequest(BaseModel):
    dir: str  # Destination directory path
    files: Optional[Dict[str, Any]] = None  # Full VFS files map: { rel_path: content }
    graphs: Optional[Dict[str, Any]] = None  # Fallback for legacy calls
    python: Optional[Dict[str, str]] = None  # Fallback for legacy calls
    python_dir: Optional[str] = None  # Optional override for the python/ source dir


@app.post("/api/state/save")
def save_state(request: StateSaveRequest):
    """
    Mirrors the browser VFS state to the requested directory, pruning redundant files.
    """
    try:
        if request.files is not None:
            out_path = storage.mirror_vfs_state(dest_dir=request.dir, files=request.files)
        else:
            out_path = storage.save_state_bundle(
                dest_dir=request.dir,
                graphs=request.graphs,
                python=request.python,
                python_dir=request.python_dir,
            )
        return {"ok": True, "path": out_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/state/load")
def load_state(dir: str = Query(..., description="Directory to load state from")):
    """
    Reads all files from the given directory and returns:
    { ok: True, files: { [rel_path]: content }, graphs: {...}, python: {...} }
    """
    try:
        data = storage.load_vfs_state(dir)
        # Also populate graphs and python keys for backwards compatibility
        graphs: Dict[str, Any] = {}
        python: Dict[str, str] = {}
        for rel_path, content in data.get("files", {}).items():
            if rel_path.endswith(".arch"):
                # Clean file_id relative to graphs/
                file_id = rel_path[:-5]
                if file_id.startswith("graphs/"):
                    file_id = file_id[7:]
                graphs[file_id] = content
            elif rel_path.endswith(".py"):
                python_rel = rel_path[7:] if rel_path.startswith("python/") else rel_path
                python[python_rel] = content

        return {"ok": True, "files": data.get("files", {}), "graphs": graphs, "python": python}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Directory not found or empty: {dir}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
