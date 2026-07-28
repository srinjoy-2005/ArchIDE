from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import BlockDef, CompileRequest, CheckRequest
from blocks import get_all_block_defs
from compiler import topological_sort, generate_pytorch_code, shape_inference_pass, ShapeError

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
    return get_all_block_defs()

@app.post("/api/compile")
def compile_graph(request: CompileRequest):
    try:
        sorted_nodes = topological_sort(request.nodes, request.edges)
        code = generate_pytorch_code(sorted_nodes, request.edges)
        return {"code": code}
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
        sorted_nodes = topological_sort(request.nodes, request.edges)
        node_shapes, node_params = shape_inference_pass(sorted_nodes, request.edges)
        # Convert tuples to lists for JSON serialisation
        serialisable = {
            node_id: {port: list(shape) for port, shape in ports.items()}
            for node_id, ports in node_shapes.items()
        }
        return {"ok": True, "node_shapes": serialisable, "node_params": node_params}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
