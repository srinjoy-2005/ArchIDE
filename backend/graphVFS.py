import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

from storage import storage

router = APIRouter()

class SaveRequest(BaseModel):
    file_id: str
    content: Dict[str, Any]

@router.get("/api/vfs/init")
def get_all_files_endpoint():
    return {"files": storage.get_all_graphs()}

@router.post("/api/vfs/save")
def save_file(req: SaveRequest):
    success = storage.save_graph(req.file_id, req.content)
    return {"ok": success}

async def file_watcher(request: Request):
    last_mtimes = storage.get_graph_mtimes()
            
    while True:
        if await request.is_disconnected():
            break
        await asyncio.sleep(0.5)
        
        current_mtimes = storage.get_graph_mtimes()
        
        for rel_path, mtime in current_mtimes.items():
            if rel_path not in last_mtimes or mtime > last_mtimes[rel_path]:
                last_mtimes[rel_path] = mtime
                file_id = rel_path[:-5] if rel_path.endswith(".arch") else rel_path
                try:
                    content = storage.read_graph(rel_path)
                    yield f"data: {json.dumps({'file_id': file_id, 'content': content})}\n\n"
                except Exception:
                    pass

@router.get("/api/vfs/stream")
async def vfs_stream(request: Request):
    return StreamingResponse(file_watcher(request), media_type="text/event-stream")
