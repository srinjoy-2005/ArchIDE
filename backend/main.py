from fastapi import HTTPException
from fastapi import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import BlockDef, CompileRequest
from registry import REGISTRY
from compiler import topological_sort, generate_pytorch_code

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/blocks", response_model=List[BlockDef])
def get_blocks():
    return REGISTRY

@app.post("/api/compile")
def compile_graph(request: CompileRequest):
    try:
        sorted_nodes = topological_sort(request.nodes,request.edges)
        code = generate_pytorch_code(sorted_nodes,request.edges)
        return {"code":code}
    except ValueError as e:
        raise HTTPException(status_code = 400,detail = str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
