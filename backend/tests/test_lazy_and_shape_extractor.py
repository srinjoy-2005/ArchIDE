import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compiler import topological_sort, generate_pytorch_code, shape_inference_multi_graph
from models import Node, Edge, NodeData, GraphData, ArchVariableModel
from blocks.core import LinearBlock, Conv2DBlock, InputBlock, ShapeExtractorBlock
from blocks import get_block_by_id


def test_linear_block_lazy_inference_and_emit():
    block = LinearBlock()

    # 1. infer_shapes with in_features="LAZY" should succeed even if input last dim doesn't match a default
    out = block.infer_shapes({"in": (16, 512)}, {"in_features": "LAZY", "out_features": 64})
    assert out["out"] == (16, 64)

    # 2. emit_init with in_features="LAZY"
    init_code = block.emit_init("fc_lazy", {"in_features": "LAZY", "out_features": 64, "bias": True})
    assert init_code == "self.layer_fc_lazy = nn.LazyLinear(64, bias=True)"

    # 3. emit_init with in_features="LAZY" and bias=False
    init_code_nobias = block.emit_init("fc_lazy", {"in_features": "LAZY", "out_features": 128, "bias": False})
    assert init_code_nobias == "self.layer_fc_lazy = nn.LazyLinear(128, bias=False)"


def test_linear_block_lazy_codegen():
    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(2, 100)"})),
        Node(id="n2", data=NodeData(block_id="linear", label="LazyLinear", paramValues={"in_features": "LAZY", "out_features": 50})),
        Node(id="n3", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n3", targetHandle="in")
    ]
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files, _, _ = generate_pytorch_code(graphs, "main")
    code = files["main"]

    assert "self.layer_n2 = nn.LazyLinear(50, bias=True)" in code
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None


def test_conv2d_block_lazy_inference_and_emit():
    block = Conv2DBlock()

    # 1. infer_shapes with in_channels="LAZY"
    params = {
        "in_channels": "LAZY",
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1
    }
    out = block.infer_shapes({"in": (4, 128, 32, 32)}, params)
    assert out["out"] == (4, 32, 32, 32)

    # 2. emit_init with in_channels="LAZY"
    init_code = block.emit_init("conv_lazy", {
        "in_channels": "LAZY",
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True
    })
    assert init_code == "self.layer_conv_lazy = nn.LazyConv2d(32, 3, stride=2, padding=1, dilation=1, groups=1, bias=True)"


def test_conv2d_block_lazy_codegen():
    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(1, 3, 64, 64)"})),
        Node(id="n2", data=NodeData(block_id="conv2d", label="LazyConv", paramValues={
            "in_channels": "LAZY",
            "out_channels": 16,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "dilation": 1,
            "groups": 1,
            "bias": False
        })),
        Node(id="n3", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n3", targetHandle="in")
    ]
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files, _, _ = generate_pytorch_code(graphs, "main")
    code = files["main"]

    assert "self.layer_n2 = nn.LazyConv2d(16, 5, stride=1, padding=2, dilation=1, groups=1, bias=False)" in code
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None


def test_shape_extractor_block_definition_and_inference():
    block = ShapeExtractorBlock()
    defn = block.definition
    assert defn.id == "shape_extractor"
    assert defn.name == "Shape Extractor"
    assert defn.category == "Core Layers"
    assert defn.color == "#a855f7"
    assert defn.is_functional is True
    assert len(defn.inputs) == 1
    assert defn.inputs[0].id == "in"

    output_ids = [p.id for p in defn.outputs]
    assert output_ids == ["shape", "dim_0", "dim_1", "dim_2", "dim_3"]

    # Inferred shapes with concrete 4D tensor
    out = block.infer_shapes({"in": (2, 3, 224, 224)}, {})
    assert out["shape"] == (4,)
    assert out["dim_0"] == (1,)
    assert out["dim_1"] == (1,)
    assert out["dim_2"] == (1,)
    assert out["dim_3"] == (1,)

    # ANY shape
    out_any = block.infer_shapes({"in": ("ANY",)}, {})
    assert out_any["shape"] == ("ANY",)
    assert out_any["dim_0"] == (1,)

    # emit_init should be empty
    assert block.emit_init("se", {}) == ""


def test_shape_extractor_forward_codegen():
    block = ShapeExtractorBlock()
    input_vars = {"in": "x_in"}
    output_vars = {
        "shape": "x_shape",
        "dim_0": "b",
        "dim_1": "c",
        "dim_2": "h",
        "dim_3": "w"
    }
    fwd = block.emit_forward("se1", input_vars, output_vars, {})
    assert "x_shape = x_in.shape" in fwd
    assert "b = x_in.shape[0] if len(x_in.shape) > 0 else None" in fwd
    assert "c = x_in.shape[1] if len(x_in.shape) > 1 else None" in fwd
    assert "h = x_in.shape[2] if len(x_in.shape) > 2 else None" in fwd
    assert "w = x_in.shape[3] if len(x_in.shape) > 3 else None" in fwd

    # End-to-end graph test with ShapeExtractor
    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(2, 3, 32, 32)"})),
        Node(id="n2", data=NodeData(block_id="shape_extractor", label="ShapeExtractor", paramValues={})),
        Node(id="n3", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="shape", target="n3", targetHandle="in")
    ]
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges)}
    files, _, _ = generate_pytorch_code(graphs, "main")
    code = files["main"]

    assert ".shape" in code
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None


def test_input_block_shape_param_and_embedded_variable():
    block = InputBlock()
    assert block.definition.params[0].type == "shape"

    nodes = [
        Node(id="n1", data=NodeData(block_id="input", label="Input", paramValues={"shape": "(1, @var:channels, 32, 32)"})),
        Node(id="n2", data=NodeData(block_id="conv2d", label="Conv", paramValues={"in_channels": -1, "out_channels": 16})),
        Node(id="n3", data=NodeData(block_id="output", label="Output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="n1", sourceHandle="out", target="n2", targetHandle="in"),
        Edge(id="e2", source="n2", sourceHandle="out", target="n3", targetHandle="in")
    ]
    variables = [
        ArchVariableModel(name="channels", type="int", default=64, scope="init_param")
    ]

    # Shape inference should not throw ShapeError and should substitute channels=64
    all_shapes, all_params = shape_inference_multi_graph(
        {"main": GraphData(name="Main", nodes=nodes, edges=edges, variables=variables)},
        "main"
    )
    assert all_shapes["n1"]["out"] == (1, 64, 32, 32)
    # n2 should have inferred in_channels=64
    assert all_params["n2"]["in_channels"] == 64

    # Code generation
    graphs = {"main": GraphData(name="Main", nodes=nodes, edges=edges, variables=variables)}
    files, _, _ = generate_pytorch_code(graphs, "main")
    code = files["main"]

    assert "self.layer_n2 = nn.Conv2d(64, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)" in code
    compiled = compile(code, "<string>", "exec")
    assert compiled is not None
