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

@pytest.mark.anyio
async def test_save_and_load_state(tmp_path):
    target_dir = str(tmp_path / "saved_state")
    sample_graphs = {
        "main": {"name": "main", "nodes": [{"id": "n1"}], "edges": []},
        "sub/block": {"name": "block", "nodes": [], "edges": []}
    }
    sample_python = {
        "main.py": "import torch\nprint('hello')",
        "sub/block.py": "import torch\nclass Block: pass"
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Save state
        save_res = await client.post("/api/state/save", json={
            "dir": target_dir,
            "graphs": sample_graphs,
            "python": sample_python
        })
        assert save_res.status_code == 200
        assert save_res.json()["ok"] is True

        # Verify files on disk
        assert (tmp_path / "saved_state" / "graphs" / "main.arch").exists()
        assert (tmp_path / "saved_state" / "graphs" / "sub" / "block.arch").exists()
        assert (tmp_path / "saved_state" / "python" / "main.py").exists()
        assert (tmp_path / "saved_state" / "python" / "sub" / "block.py").exists()
        assert (tmp_path / "saved_state" / "state.archstate").exists()

        # Load state
        load_res = await client.get(f"/api/state/load?dir={target_dir}")
        assert load_res.status_code == 200
        loaded_data = load_res.json()
        assert loaded_data["ok"] is True
        assert "main" in loaded_data["graphs"]
        assert "sub/block" in loaded_data["graphs"]
        assert "main.py" in loaded_data["python"]
        assert "sub/block.py" in loaded_data["python"]
        assert "print('hello')" in loaded_data["python"]["main.py"]

@pytest.mark.anyio
async def test_mirror_vfs_state_and_prune(tmp_path):
    target_dir = str(tmp_path / "mirrored_workspace")
    initial_vfs = {
        "archide.toml": '[directories]\ngraphs_dir = "graphs"',
        "graphs/main.arch": {"name": "main", "nodes": [{"id": "n1"}], "edges": []},
        "graphs/sub.arch": {"name": "sub", "nodes": [{"id": "n2"}], "edges": []},
        "python/main.py": "import torch\n# main",
        "python/sub.py": "import torch\n# sub",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Initial save: all 5 files created
        res1 = await client.post("/api/state/save", json={
            "dir": target_dir,
            "files": initial_vfs
        })
        assert res1.status_code == 200
        assert (tmp_path / "mirrored_workspace" / "archide.toml").exists()
        assert (tmp_path / "mirrored_workspace" / "graphs" / "main.arch").exists()
        assert (tmp_path / "mirrored_workspace" / "graphs" / "sub.arch").exists()
        assert (tmp_path / "mirrored_workspace" / "python" / "main.py").exists()
        assert (tmp_path / "mirrored_workspace" / "python" / "sub.py").exists()

        # 2. Second save: user deleted "sub.arch" and "sub.py" in browser
        updated_vfs = {
            "archide.toml": '[directories]\ngraphs_dir = "graphs"',
            "graphs/main.arch": {"name": "main", "nodes": [{"id": "n1"}], "edges": []},
            "python/main.py": "import torch\n# main",
        }
        res2 = await client.post("/api/state/save", json={
            "dir": target_dir,
            "files": updated_vfs
        })
        assert res2.status_code == 200

        # Verify redundant files were PRUNED from disk
        assert (tmp_path / "mirrored_workspace" / "archide.toml").exists()
        assert (tmp_path / "mirrored_workspace" / "graphs" / "main.arch").exists()
        assert not (tmp_path / "mirrored_workspace" / "graphs" / "sub.arch").exists()
        assert (tmp_path / "mirrored_workspace" / "python" / "main.py").exists()
        assert not (tmp_path / "mirrored_workspace" / "python" / "sub.py").exists()

        # 3. Load state: verify exact files are returned
        load_res = await client.get(f"/api/state/load?dir={target_dir}")
        assert load_res.status_code == 200
        loaded = load_res.json()
        assert "files" in loaded
        assert "archide.toml" in loaded["files"]
        assert "graphs/main.arch" in loaded["files"]
        assert "graphs/sub.arch" not in loaded["files"]
        assert "python/main.py" in loaded["files"]
        assert "python/sub.py" not in loaded["files"]


