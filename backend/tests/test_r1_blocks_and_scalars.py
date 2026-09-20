import math
import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blocks.activations import GELUBlock, SiLUBlock
from blocks.core import Conv1DBlock, EmbeddingBlock
from blocks.tensor_ops import AddBlock, SubBlock, MulBlock, DivBlock
from blocks import get_block_by_id, get_all_blocks
from compiler import generate_pytorch_code
from python_decompiler import decompile_python_to_ir
from models import GraphData, Node, Edge, NodeData


# ==============================================================================
# 1. GELUBlock Tests
# ==============================================================================

def test_gelu_block_definition():
    block = GELUBlock()
    assert block.definition.id == "gelu"
    assert block.definition.name == "GELU"
    assert block.definition.category == "Activations"
    assert not block.definition.is_functional
    assert len(block.definition.inputs) == 1
    assert len(block.definition.outputs) == 1


def test_gelu_block_shape_inference():
    block = GELUBlock()
    assert block.infer_shapes({"in": (2, 64, 128)}, {}) == {"out": (2, 64, 128)}
    assert block.infer_shapes({"in": ("ANY",)}, {}) == {"out": ("ANY",)}
    assert block.infer_shapes({}, {}) == {"out": ("ANY",)}


def test_gelu_block_emit():
    block = GELUBlock()
    init_none = block.emit_init("gelu_1", {"approximate": "none"})
    assert "nn.GELU(approximate='none')" in init_none

    init_tanh = block.emit_init("gelu_1", {"approximate": "tanh"})
    assert "nn.GELU(approximate='tanh')" in init_tanh

    fwd = block.emit_forward("gelu_1", {"in": "x"}, {"out": "out_x"}, {})
    assert fwd == "out_x = self.layer_gelu_1(x)"


# ==============================================================================
# 2. SiLUBlock Tests
# ==============================================================================

def test_silu_block_definition():
    block = SiLUBlock()
    assert block.definition.id == "silu"
    assert block.definition.name == "SiLU"
    assert block.definition.category == "Activations"
    assert not block.definition.is_functional
    assert len(block.definition.inputs) == 1
    assert len(block.definition.outputs) == 1


def test_silu_block_shape_inference():
    block = SiLUBlock()
    assert block.infer_shapes({"in": (4, 32, 64)}, {}) == {"out": (4, 32, 64)}
    assert block.infer_shapes({"in": ("ANY",)}, {}) == {"out": ("ANY",)}


def test_silu_block_emit():
    block = SiLUBlock()
    init_false = block.emit_init("silu_1", {"inplace": False})
    assert "nn.SiLU(inplace=False)" in init_false

    init_true = block.emit_init("silu_1", {"inplace": True})
    assert "nn.SiLU(inplace=True)" in init_true

    fwd = block.emit_forward("silu_1", {"in": "x"}, {"out": "out_x"}, {})
    assert fwd == "out_x = self.layer_silu_1(x)"


# ==============================================================================
# 3. Conv1DBlock Tests
# ==============================================================================

def test_conv1d_block_definition():
    block = Conv1DBlock()
    assert block.definition.id == "conv1d"
    assert block.definition.name == "Conv1D"
    assert block.definition.category == "Core Layers"
    assert not block.definition.is_functional


def test_conv1d_block_shape_inference():
    block = Conv1DBlock()
    # (B, C, L) = (2, 3, 100), k=3, s=2, pad=1 -> out_l = floor((100 + 2*1 - 1*(3-1) - 1)/2 + 1) = floor(99/2 + 1) = 50
    res = block.infer_shapes(
        {"in": (2, 3, 100)},
        {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1, "dilation": 1}
    )
    assert res == {"out": (2, 16, 50)}

    # Auto-infer in_channels from input shape when in_channels == -1
    params = {"in_channels": -1, "out_channels": 8, "kernel_size": 1, "stride": 1, "padding": 0}
    res2 = block.infer_shapes({"in": (4, 12, 50)}, params)
    assert res2 == {"out": (4, 8, 50)}
    assert params["in_channels"] == 12

    # LAZY in_channels
    res_lazy = block.infer_shapes(
        {"in": (4, 12, 50)},
        {"in_channels": "LAZY", "out_channels": 8, "kernel_size": 1, "stride": 1, "padding": 0}
    )
    assert res_lazy == {"out": (4, 8, 50)}


def test_conv1d_block_errors():
    block = Conv1DBlock()
    # Non-3D tensor
    with pytest.raises(ValueError, match="Conv1D expects a 3D tensor"):
        block.infer_shapes({"in": (2, 3, 32, 32)}, {"in_channels": 3})

    # Channel mismatch
    with pytest.raises(ValueError, match="expected in_channels=4, but input has 3"):
        block.infer_shapes({"in": (2, 3, 32)}, {"in_channels": 4, "out_channels": 8})

    # Invalid stride
    with pytest.raises(ValueError, match="stride must be greater than 0"):
        block.infer_shapes({"in": (2, 3, 32)}, {"in_channels": 3, "stride": 0})

    # Invalid kernel_size
    with pytest.raises(ValueError, match="kernel_size must be greater than 0"):
        block.infer_shapes({"in": (2, 3, 32)}, {"in_channels": 3, "kernel_size": -1})

    # Negative output length
    with pytest.raises(ValueError, match="Negative spatial dimension length"):
        block.infer_shapes({"in": (2, 3, 2)}, {"in_channels": 3, "kernel_size": 10, "stride": 1, "padding": 0})


def test_conv1d_emit():
    block = Conv1DBlock()
    init_std = block.emit_init("c1", {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True})
    assert "nn.Conv1d(3, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)" in init_std

    init_lazy = block.emit_init("c2", {"in_channels": "LAZY", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True})
    assert "nn.LazyConv1d(16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)" in init_lazy

    fwd = block.emit_forward("c1", {"in": "x"}, {"out": "out_c"}, {})
    assert fwd == "out_c = self.layer_c1(x)"


# ==============================================================================
# 4. EmbeddingBlock Tests
# ==============================================================================

def test_embedding_block_definition():
    block = EmbeddingBlock()
    assert block.definition.id == "embedding"
    assert block.definition.name == "Embedding"
    assert block.definition.category == "Core Layers"
    assert not block.definition.is_functional


def test_embedding_block_shape_inference():
    block = EmbeddingBlock()
    # 2D input -> 3D output
    assert block.infer_shapes({"in": (4, 32)}, {"num_embeddings": 1000, "embedding_dim": 128}) == {"out": (4, 32, 128)}
    # 1D input -> 2D output
    assert block.infer_shapes({"in": (16,)}, {"num_embeddings": 500, "embedding_dim": 64}) == {"out": (16, 64)}
    # ANY input
    assert block.infer_shapes({"in": ("ANY",)}, {}) == {"out": ("ANY",)}


def test_embedding_block_errors():
    block = EmbeddingBlock()
    with pytest.raises(ValueError, match="num_embeddings must be greater than 0"):
        block.infer_shapes({"in": (4, 32)}, {"num_embeddings": 0, "embedding_dim": 128})

    with pytest.raises(ValueError, match="embedding_dim must be greater than 0"):
        block.infer_shapes({"in": (4, 32)}, {"num_embeddings": 1000, "embedding_dim": -5})


def test_embedding_emit():
    block = EmbeddingBlock()
    init_std = block.emit_init("emb_1", {"num_embeddings": 1000, "embedding_dim": 128})
    assert init_std == "self.layer_emb_1 = nn.Embedding(1000, 128)"

    init_adv = block.emit_init("emb_2", {
        "num_embeddings": 5000,
        "embedding_dim": 256,
        "padding_idx": 0,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "sparse": True
    })
    assert "nn.Embedding(5000, 256, padding_idx=0, max_norm=1.0, scale_grad_by_freq=True, sparse=True)" in init_adv

    fwd = block.emit_forward("emb_1", {"in": "indices"}, {"out": "x"}, {})
    assert fwd == "x = self.layer_emb_1(indices)"


# ==============================================================================
# 5. Scalar Binary Operations Tests (Add, Sub, Mul, Div)
# ==============================================================================

def test_add_block_scalars():
    block = AddBlock()
    # x + 1 (scalar_b)
    fwd = block.emit_forward("add_1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": "1"})
    assert fwd == "y = x + 1"

    # 1 + x (scalar_a)
    fwd = block.emit_forward("add_2", {"in": ["x"]}, {"out": "y"}, {"scalar_a": "1"})
    assert fwd == "y = 1 + x"

    # None or missing scalars
    fwd = block.emit_forward("add_3", {"in": ["x", "z"]}, {"out": "y"}, {"scalar_a": None, "scalar_b": "None"})
    assert fwd == "y = x + z"


def test_sub_block_scalars():
    block = SubBlock()
    # x - 1
    fwd = block.emit_forward("sub_1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "1"})
    assert fwd == "y = x - 1"

    # 1 - x
    fwd = block.emit_forward("sub_2", {"in_b": "x"}, {"out": "y"}, {"scalar_a": "1"})
    assert fwd == "y = 1 - x"

    # Tensor - Tensor
    fwd = block.emit_forward("sub_3", {"in_a": "x", "in_b": "z"}, {"out": "y"}, {})
    assert fwd == "y = x - z"


def test_mul_block_scalars():
    block = MulBlock()
    # x * 2
    fwd = block.emit_forward("mul_1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": "2"})
    assert fwd == "y = x * 2"

    # 2 * x
    fwd = block.emit_forward("mul_2", {"in": ["x"]}, {"out": "y"}, {"scalar_a": "2"})
    assert fwd == "y = 2 * x"


def test_div_block_scalars():
    block = DivBlock()
    # x / math.sqrt(self.d_model)
    fwd = block.emit_forward("div_1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "math.sqrt(self.d_model)"})
    assert fwd == "y = x / math.sqrt(self.d_model)"

    # 1 / x
    fwd = block.emit_forward("div_2", {"in_b": "x"}, {"out": "y"}, {"scalar_a": "1"})
    assert fwd == "y = 1 / x"


# ==============================================================================
# 6. End-to-End Compiler & PyTorch Execution Verification
# ==============================================================================

def test_compiler_generates_and_executes_new_blocks_and_scalars():
    nodes = [
        Node(id="in1", type="blockNode", data=NodeData(block_id="input", label="input", paramValues={"shape": "(2, 8)"})),
        Node(id="emb", type="blockNode", data=NodeData(block_id="embedding", label="emb", paramValues={"num_embeddings": 100, "embedding_dim": 16})),
        Node(id="tr",  type="blockNode", data=NodeData(block_id="transpose", label="tr", paramValues={"dim0": 1, "dim1": 2})),
        Node(id="c1d", type="blockNode", data=NodeData(block_id="conv1d", label="c1d", paramValues={"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1})),
        Node(id="gelu", type="blockNode", data=NodeData(block_id="gelu", label="gelu", paramValues={"approximate": "none"})),
        Node(id="silu", type="blockNode", data=NodeData(block_id="silu", label="silu", paramValues={"inplace": False})),
        Node(id="add_sc", type="blockNode", data=NodeData(block_id="add", label="add_sc", paramValues={"scalar_b": "1.5"})),
        Node(id="mul_sc", type="blockNode", data=NodeData(block_id="mul", label="mul_sc", paramValues={"scalar_b": "2.0"})),
        Node(id="div_sc", type="blockNode", data=NodeData(block_id="div", label="div_sc", paramValues={"scalar_b": "math.sqrt(4.0)"})),
        Node(id="out", type="blockNode", data=NodeData(block_id="output", label="output", paramValues={}))
    ]
    edges = [
        Edge(id="e1", source="in1", sourceHandle="out", target="emb", targetHandle="in"),
        Edge(id="e2", source="emb", sourceHandle="out", target="tr", targetHandle="in"),
        Edge(id="e3", source="tr", sourceHandle="out", target="c1d", targetHandle="in"),
        Edge(id="e4", source="c1d", sourceHandle="out", target="gelu", targetHandle="in"),
        Edge(id="e5", source="gelu", sourceHandle="out", target="silu", targetHandle="in"),
        Edge(id="e6", source="silu", sourceHandle="out", target="add_sc", targetHandle="in"),
        Edge(id="e7", source="add_sc", sourceHandle="out", target="mul_sc", targetHandle="in"),
        Edge(id="e8", source="mul_sc", sourceHandle="out", target="div_sc", targetHandle="in_a"),
        Edge(id="e9", source="div_sc", sourceHandle="out", target="out", targetHandle="in")
    ]
    graphs = {"main": GraphData(name="TestM1Model", nodes=nodes, edges=edges)}
    files, _, _ = generate_pytorch_code(graphs, "main")
    code = files["main"]

    # Verify header contains import math
    assert "import math" in code
    assert "nn.Embedding" in code
    assert "nn.Conv1d" in code
    assert "nn.GELU" in code
    assert "nn.SiLU" in code
    assert "math.sqrt(4.0)" in code

    # Execute generated code in a clean namespace
    namespace = {}
    exec(code, namespace)
    assert "Model" in namespace

    model_cls = namespace["Model"]
    model = model_cls()
    model.eval()

    # Pass integer indices tensor (2, 8)
    indices = torch.randint(0, 100, (2, 8), dtype=torch.long)
    with torch.no_grad():
        output = model(indices)

    assert output is not None
    # Shape check: (2, 8) -> Embedding (2, 8, 16) -> Transpose (2, 16, 8) -> Conv1D (2, 32, 8) -> GELU -> SiLU -> scalar math
    assert output.shape == (2, 32, 8)

    # Validate mathematical correctness of scalar math: ((silu_out + 1.5) * 2.0) / 2.0 == silu_out + 1.5
    raw_emb = model.layer_emb(indices).transpose(1, 2)
    raw_c1d = model.layer_c1d(raw_emb)
    raw_gelu = model.layer_gelu(raw_c1d)
    raw_silu = model.layer_silu(raw_gelu)
    expected = (raw_silu + 1.5) * 2.0 / math.sqrt(4.0)

    assert torch.allclose(output, expected, atol=1e-5)


# ==============================================================================
# 7. Python Decompiler Roundtrip & Mapping Tests
# ==============================================================================

def test_decompiler_handles_conv1d_embedding_gelu_silu():
    py_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(500, 64)
        self.conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act1(x)
        x = self.act2(x)
        x = x + 1.0
        x = x - 0.5
        x = x * 2.0
        x = x / 4.0
        return x
"""
    ir = decompile_python_to_ir(py_code)
    blocks_used = {n["block"] for n in ir["nodes"].values()}

    assert "embedding" in blocks_used
    assert "conv1d" in blocks_used
    assert "gelu" in blocks_used
    assert "silu" in blocks_used
    assert "add" in blocks_used
    assert "sub" in blocks_used
    assert "mul" in blocks_used
    assert "div" in blocks_used

    # Check that scalar operands are captured in node params
    add_nodes = [n for n in ir["nodes"].values() if n["block"] == "add"]
    assert any(n.get("params", {}).get("scalar_b") == 1.0 for n in add_nodes)

    sub_nodes = [n for n in ir["nodes"].values() if n["block"] == "sub"]
    assert any(n.get("params", {}).get("scalar_b") == 0.5 for n in sub_nodes)

    mul_nodes = [n for n in ir["nodes"].values() if n["block"] == "mul"]
    assert any(n.get("params", {}).get("scalar_b") == 2.0 for n in mul_nodes)

    div_nodes = [n for n in ir["nodes"].values() if n["block"] == "div"]
    assert any(n.get("params", {}).get("scalar_b") == 4.0 for n in div_nodes)
