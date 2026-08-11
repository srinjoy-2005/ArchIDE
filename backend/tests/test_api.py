import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def test_compile_shape_mismatch():
    # Construct a payload with a Multiply block trying to multiply (4, 3) and (3, 4)
    # This should fail the shape broadcasting and return 422
    payload = {
        "nodes": [
            {
                "id": "node_input_a",
                "data": {
                    "block_id": "input",
                    "label": "Input A",
                    "paramValues": {"shape": "(4, 3)"}
                }
            },
            {
                "id": "node_input_b",
                "data": {
                    "block_id": "input",
                    "label": "Input B",
                    "paramValues": {"shape": "(3, 4)"}
                }
            },
            {
                "id": "node_mul",
                "data": {
                    "block_id": "mul",
                    "label": "Multiply"
                }
            }
        ],
        "edges": [
            {
                "id": "edge_a",
                "source": "node_input_a",
                "sourceHandle": "out",
                "target": "node_mul",
                "targetHandle": "in_a"
            },
            {
                "id": "edge_b",
                "source": "node_input_b",
                "sourceHandle": "out",
                "target": "node_mul",
                "targetHandle": "in_b"
            }
        ]
    }
    
    response = client.post("/api/check", json=payload)
    
    # 422 is returned by check_graph in main.py when a ShapeError is raised
    assert response.status_code == 422
    
    data = response.json()
    assert "detail" in data
    assert "not broadcastable" in data["detail"].get("message", "")
    assert data["detail"].get("node_id") == "node_mul"
