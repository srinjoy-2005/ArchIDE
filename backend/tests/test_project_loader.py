import pytest
import sys
import os
from pathlib import Path
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_loader import load_project, compile_project, check_project

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_and_run_resnet_project(tmp_path):
    """End-to-end test: load multi-submodule ResNet, compile, and execute live PyTorch forward pass."""
    project_path = FIXTURES_DIR / "resnet_project"
    
    # 1. Load project
    req = load_project(project_path)
    assert req.main_graph_id == "main"
    assert "conv_bn_relu" in req.graphs
    assert "res_block" in req.graphs
    assert "main" in req.graphs
    
    # 2. Check shapes
    check_result = check_project(project_path)
    assert "node_shapes" in check_result
    
    # 3. Compile to PyTorch code
    files, _, _ = compile_project(project_path)
    code = "\n".join(files.values())
    assert "class ConvBnRelu(nn.Module):" in code
    assert "class ResBlock(nn.Module):" in code
    assert "class Model(nn.Module):" in code
    
    # 3. Live PyTorch execution
    import sys, os
    import importlib.util
    for filename, filecode in files.items():
        with open(tmp_path / f"{filename}.py", "w") as f:
            f.write(filecode)
    
    sys.path.insert(0, str(tmp_path))
    try:
        spec = importlib.util.spec_from_file_location("main", tmp_path / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.Model()
        import torch
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)
    finally:
        sys.path.pop(0)


def test_load_and_run_transformer_project(tmp_path):
    """End-to-end test: load Transformer encoder with MLP submodule, constructor params, and live execution."""
    project_path = FIXTURES_DIR / "transformer_project"
    
    # 1. Load project
    req = load_project(project_path)
    assert req.main_graph_id == "main"
    assert "mlp_block" in req.graphs
    
    # 2. Compile to PyTorch code
    files, _, _ = compile_project(project_path)
    code = "\n".join(files.values())
    assert "class MlpBlock(nn.Module):" in code
    assert "def __init__(self, d_model: int = 64, d_ff: int = 128):" in code
    assert "self.custom_t_mlp = MlpBlock(d_model=64, d_ff=128)" in code
    
    # 3. Live PyTorch execution
    import sys, os
    import importlib.util
    for filename, filecode in files.items():
        with open(tmp_path / f"{filename}.py", "w") as f:
            f.write(filecode)
    
    sys.path.insert(0, str(tmp_path))
    try:
        spec = importlib.util.spec_from_file_location("main", tmp_path / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.Model()
        import torch
        x = torch.randn(1, 16, 64)
        out = model(x)
        assert out.shape == (1, 16, 64)
    finally:
        sys.path.pop(0)


def test_single_file_project_loader(tmp_path):
    """Test loading directly from an individual JSON file."""
    graph_json = tmp_path / "simple_model.json"
    graph_json.write_text("""{
      "name": "SimpleNet",
      "nodes": [
        {"id": "n1", "data": {"block_id": "input", "label": "x", "paramValues": {"shape": "(1, 10)"}}},
        {"id": "n2", "data": {"block_id": "linear", "label": "FC", "paramValues": {"in_features": 10, "out_features": 2}}},
        {"id": "n3", "data": {"block_id": "output", "label": "out", "paramValues": {}}}
      ],
      "edges": [
        {"id": "e1", "source": "n1", "sourceHandle": "out", "target": "n2", "targetHandle": "in"},
        {"id": "e2", "source": "n2", "sourceHandle": "out", "target": "n3", "targetHandle": "in"}
      ]
    }""", encoding="utf-8")
    
    req = load_project(graph_json)
    assert "simple_model" in req.graphs
    files, _, _ = compile_project(graph_json)
    code = "\n".join(files.values())
    
    namespace = {}
    exec(code, namespace)
    model = namespace["Model"]()
    out = model(torch.randn(1, 10))
    assert out.shape == (1, 2)
