import pytest
import httpx
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

def wrap_payload(nodes, edges):
    return {
        "main_graph_id": "main",
        "graphs": {
            "main": {
                "name": "Main",
                "nodes": nodes,
                "edges": edges
            }
        }
    }

@pytest.mark.anyio
async def test_compile_shape_mismatch():
    nodes = [
        {"id": "node_input_a", "data": {"block_id": "input", "label": "Input A", "paramValues": {"shape": "(4, 3)"}}},
        {"id": "node_input_b", "data": {"block_id": "input", "label": "Input B", "paramValues": {"shape": "(3, 4)"}}},
        {"id": "node_mul", "data": {"block_id": "mul", "label": "Multiply", "paramValues": {}}}
    ]
    edges = [
        {"id": "edge_a", "source": "node_input_a", "sourceHandle": "out", "target": "node_mul", "targetHandle": "in"},
        {"id": "edge_b", "source": "node_input_b", "sourceHandle": "out", "target": "node_mul", "targetHandle": "in"}
    ]
    payload = wrap_payload(nodes, edges)
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/check", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "not broadcastable" in data["detail"].get("message", "")
        assert data["detail"].get("node_id") == "node_mul"

@pytest.mark.anyio
async def test_get_blocks():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/blocks")
        assert response.status_code == 200
        blocks = response.json()
        assert isinstance(blocks, list)
        assert len(blocks) > 0
        assert any(b["id"] == "linear" for b in blocks)

@pytest.mark.anyio
async def test_compile_cycle():
    nodes = [
        {"id": "n1", "data": {"block_id": "linear", "label": "A", "paramValues": {}}},
        {"id": "n2", "data": {"block_id": "linear", "label": "B", "paramValues": {}}}
    ]
    edges = [
        {"id": "e1", "source": "n1", "sourceHandle": "out", "target": "n2", "targetHandle": "in"},
        {"id": "e2", "source": "n2", "sourceHandle": "out", "target": "n1", "targetHandle": "in"}
    ]
    payload = wrap_payload(nodes, edges)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/compile", json=payload)
        assert response.status_code == 400
        assert "Cycle detected" in response.json().get("detail", "")

@pytest.mark.anyio
async def test_compile_success():
    nodes = [
        {"id": "n1", "data": {"block_id": "input", "label": "Input", "paramValues": {"shape": "(2, 10)"}}},
        {"id": "n2", "data": {"block_id": "linear", "label": "Linear", "paramValues": {"in_features": 10, "out_features": 5}}},
        {"id": "n3", "data": {"block_id": "output", "label": "Output", "paramValues": {}}}
    ]
    edges = [
        {"id": "e1", "source": "n1", "sourceHandle": "out", "target": "n2", "targetHandle": "in"},
        {"id": "e2", "source": "n2", "sourceHandle": "out", "target": "n3", "targetHandle": "in"}
    ]
    payload = wrap_payload(nodes, edges)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/compile", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "nn.Linear" in data["files"]["main"]
