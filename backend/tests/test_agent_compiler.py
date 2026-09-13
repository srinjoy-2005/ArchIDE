import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent_compiler import AgentGraphCompiler, compile_ir
from compiler import ShapeError

def test_compile_basic_mlp():
    ir = {
        "name": "BasicMLP",
        "variables": [{"name": "hidden_dim", "type": "int", "default": 128}],
        "nodes": {
            "in": {"block": "input", "params": {"shape": "(1, 784)"}},
            "fc1": {"block": "linear", "params": {"in_features": 784, "out_features": 128}},
            "act": {"block": "relu"},
            "fc2": {"block": "linear", "params": {"in_features": 128, "out_features": 10}},
            "out": {"block": "output"}
        },
        "edges": [
            "in.out -> fc1.in",
            "fc1.out -> act.in",
            "act.out -> fc2.in",
            "fc2.out -> out.in"
        ]
    }

    compiler = AgentGraphCompiler(ir)
    graph_dict = compiler.compile()

    assert graph_dict["name"] == "BasicMLP"
    assert len(graph_dict["nodes"]) == 5
    assert len(graph_dict["edges"]) == 4

    # Verify positions are ordered along X
    nodes_by_block = {n["data"]["block_id"]: n["position"] for n in graph_dict["nodes"]}
    assert nodes_by_block["input"]["x"] < nodes_by_block["relu"]["x"]
    assert nodes_by_block["relu"]["x"] < nodes_by_block["output"]["x"]

    # Verify validation pass
    val_result = compiler.validate(graph_dict)
    assert val_result["valid"] is True
    assert "class Model(nn.Module):" in val_result["code"]

def test_compile_residual_branch():
    ir = {
        "name": "ResidualBlock",
        "nodes": {
            "in": {"block": "input", "params": {"shape": "(1, 64, 32, 32)"}},
            "conv1": {"block": "conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            "relu1": {"block": "relu"},
            "conv2": {"block": "conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            "add": {"block": "add"},
            "relu2": {"block": "relu"},
            "out": {"block": "output"}
        },
        "edges": [
            "in.out -> conv1.in",
            "conv1.out -> relu1.in",
            "relu1.out -> conv2.in",
            "conv2.out -> add.a",
            "in.out -> add.b",
            "add.out -> relu2.in",
            "relu2.out -> out.in"
        ]
    }

    graph = compile_ir(ir, validate=True)
    assert len(graph["nodes"]) == 7
    assert len(graph["edges"]) == 7

def test_shape_mismatch_error_detection():
    ir = {
        "name": "BrokenGraph",
        "nodes": {
            "in": {"block": "input", "params": {"shape": "(1, 100)"}},
            "fc1": {"block": "linear", "params": {"in_features": 100, "out_features": 50}},
            # Mismatch: fc2 expects in_features=20, but gets 50
            "fc2": {"block": "linear", "params": {"in_features": 20, "out_features": 10}},
            "out": {"block": "output"}
        },
        "edges": [
            "in.out -> fc1.in",
            "fc1.out -> fc2.in",
            "fc2.out -> out.in"
        ]
    }

    compiler = AgentGraphCompiler(ir)
    graph_dict = compiler.compile()

    with pytest.raises(ShapeError) as exc_info:
        compiler.validate(graph_dict)

    assert "Linear: expected in_features=20, but input last dim is 50" in str(exc_info.value)
