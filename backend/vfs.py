import os
import json
import asyncio
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()
WORKSPACE_DIR = Path(__file__).parent.parent / "workspace" / "graphs"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

class SaveRequest(BaseModel):
    file_id: str
    content: Dict[str, Any]

def get_all_arch_files():
    """Recursively finds all .arch files in WORKSPACE_DIR and returns dict mapping relative path -> content"""
    files = {}
    if os.path.exists(WORKSPACE_DIR):
        for filepath in WORKSPACE_DIR.rglob("*.arch"):
            rel_path = filepath.relative_to(WORKSPACE_DIR)
            # e.g. "conv/res_block.arch" -> "conv/res_block"
            file_id = str(rel_path.with_suffix("")).replace("\\", "/")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    files[file_id] = json.load(f)
            except Exception:
                pass
    return files

@router.get("/api/vfs/init")
def get_all_files_endpoint():
    return {"files": get_all_arch_files()}

@router.post("/api/vfs/save")
def save_file(req: SaveRequest):
    filepath = WORKSPACE_DIR / f"{req.file_id}.arch"
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(req.content, f, indent=2)
    return {"ok": True}

async def file_watcher(request: Request):
    last_mtimes = {}
    
    # Init
    if os.path.exists(WORKSPACE_DIR):
        for filepath in WORKSPACE_DIR.rglob("*.arch"):
            rel_path = str(filepath.relative_to(WORKSPACE_DIR)).replace("\\", "/")
            last_mtimes[rel_path] = os.path.getmtime(filepath)
            
    while True:
        if await request.is_disconnected():
            break
        await asyncio.sleep(0.5)
        
        if not os.path.exists(WORKSPACE_DIR):
            continue
            
        # Recursive glob to find current files
        current_files = list(WORKSPACE_DIR.rglob("*.arch"))
        current_rel_paths = {str(fp.relative_to(WORKSPACE_DIR)).replace("\\", "/"): fp for fp in current_files}
        
        for rel_path, filepath in current_rel_paths.items():
            try:
                mtime = os.path.getmtime(filepath)
            except FileNotFoundError:
                continue
                
            if rel_path not in last_mtimes or mtime > last_mtimes[rel_path]:
                last_mtimes[rel_path] = mtime
                file_id = rel_path[:-5] if rel_path.endswith(".arch") else rel_path
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        yield f"data: {json.dumps({'file_id': file_id, 'content': content})}\n\n"
                except Exception:
                    pass

@router.get("/api/vfs/stream")
async def vfs_stream(request: Request):
    return StreamingResponse(file_watcher(request), media_type="text/event-stream")
