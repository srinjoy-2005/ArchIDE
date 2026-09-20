"""
Adversarial Stress Test Harness for Milestone M1
Empirically challenges GELU, SiLU, Conv1D, Embedding, and Scalar Binary Operations.
"""

import math
import pytest
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blocks import get_block_by_id
from blocks.activations import GELUBlock, SiLUBlock
from blocks.core import Conv1DBlock, EmbeddingBlock, parse_int_1d
from blocks.tensor_ops import AddBlock, SubBlock, MulBlock, DivBlock, _is_valid_scalar
from compiler import generate_pytorch_code
from models import GraphData, Node, NodeData, Edge, ArchVariableModel


# =============================================================================
# 1. GELUBlock Adversarial Tests
# =============================================================================

def test_gelu_adversarial_approximate_modes():
    gelu = GELUBlock()
    # Check default
    assert "approximate='none'" in gelu.emit_init("1", {})
    # Check tanh case variations
    assert "approximate='tanh'" in gelu.emit_init("2", {"approximate": "tanh"})
    assert "approximate='tanh'" in gelu.emit_init("3", {"approximate": "TANH"})
    assert "approximate='tanh'" in gelu.emit_init("4", {"approximate": "'tanh'"})
    assert "approximate='tanh'" in gelu.emit_init("5", {"approximate": '"tanh"'})
    # Check invalid/unknown approximate mode gracefully falls back to none
    assert "approximate='none'" in gelu.emit_init("6", {"approximate": "invalid_mode"})
    assert "approximate='none'" in gelu.emit_init("7", {"approximate": None})

def test_gelu_arbitrary_dimensions_and_numerical_precision():
    gelu = GELUBlock()
    for shape in [(8,), (2, 8), (2, 4, 8), (2, 3, 4, 8)]:
        out_shape = gelu.infer_shapes({"in": shape}, {})["out"]
        assert out_shape == shape

        # Test PyTorch numerical execution
        layer_none = nn.GELU(approximate="none")
        layer_tanh = nn.GELU(approximate="tanh")
        x = torch.randn(*shape)
        assert torch.allclose(layer_none(x), torch.nn.functional.gelu(x, approximate="none"), atol=1e-6)
        assert torch.allclose(layer_tanh(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


# =============================================================================
# 2. SiLUBlock Adversarial Tests
# =============================================================================

def test_silu_adversarial_inplace_modes():
    silu = SiLUBlock()
    assert "inplace=False" in silu.emit_init("1", {})
    assert "inplace=True" in silu.emit_init("2", {"inplace": True})
    assert "inplace=False" in silu.emit_init("3", {"inplace": False})

def test_silu_numerical_precision():
    silu = SiLUBlock()
    for shape in [(16,), (4, 16), (2, 4, 16), (1, 2, 3, 4)]:
        out_shape = silu.infer_shapes({"in": shape}, {})["out"]
        assert out_shape == shape

        layer = nn.SiLU()
        x = torch.randn(*shape)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(layer(x), expected, atol=1e-6)


# =============================================================================
# 3. Conv1DBlock Adversarial Tests
# =============================================================================

def test_conv1d_rejects_non_3d_input():
    conv = Conv1DBlock()
    # 1D input
    with pytest.raises(ValueError, match="Conv1D expects a 3D tensor"):
        conv.infer_shapes({"in": (32,)}, {"in_channels": 3, "out_channels": 16})
    # 2D input
    with pytest.raises(ValueError, match="Conv1D expects a 3D tensor"):
        conv.infer_shapes({"in": (2, 32)}, {"in_channels": 3, "out_channels": 16})
    # 4D input
    with pytest.raises(ValueError, match="Conv1D expects a 3D tensor"):
        conv.infer_shapes({"in": (2, 3, 32, 32)}, {"in_channels": 3, "out_channels": 16})

def test_conv1d_rejects_invalid_parameters():
    conv = Conv1DBlock()
    in_s = {"in": (2, 8, 32)}

    # Stride <= 0
    with pytest.raises(ValueError, match="stride must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "stride": 0})
    with pytest.raises(ValueError, match="stride must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "stride": -1})

    # Kernel size <= 0
    with pytest.raises(ValueError, match="kernel_size must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "kernel_size": 0})
    with pytest.raises(ValueError, match="kernel_size must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "kernel_size": -3})

    # Dilation <= 0
    with pytest.raises(ValueError, match="dilation must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "dilation": 0})
    with pytest.raises(ValueError, match="dilation must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "dilation": -2})

    # Padding < 0
    with pytest.raises(ValueError, match="padding must be non-negative"):
        conv.infer_shapes(in_s, {"in_channels": 8, "padding": -1})

    # Groups <= 0
    with pytest.raises(ValueError, match="groups must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "groups": 0})
    with pytest.raises(ValueError, match="groups must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "groups": -1})

    # in_channels <= 0
    with pytest.raises(ValueError, match="in_channels must be greater than 0"):
        conv.infer_shapes({"in": (2, "ANY", 32)}, {"in_channels": 0})
    with pytest.raises(ValueError, match="in_channels must be greater than 0"):
        conv.infer_shapes({"in": (2, "ANY", 32)}, {"in_channels": -2})

    # out_channels <= 0
    with pytest.raises(ValueError, match="out_channels must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "out_channels": 0})
    with pytest.raises(ValueError, match="out_channels must be greater than 0"):
        conv.infer_shapes(in_s, {"in_channels": 8, "out_channels": -4})

    # Divisibility by groups
    with pytest.raises(ValueError, match="in_channels .* must be divisible by groups"):
        conv.infer_shapes(in_s, {"in_channels": 8, "groups": 3})
    with pytest.raises(ValueError, match="out_channels .* must be divisible by groups"):
        conv.infer_shapes(in_s, {"in_channels": 8, "out_channels": 10, "groups": 4})

def test_conv1d_rejects_negative_spatial_dimension():
    conv = Conv1DBlock()
    # L=4, kernel_size=5, stride=1, padding=0, dilation=1 -> 4 - 4 = 0 <= 0
    with pytest.raises(ValueError, match="Negative spatial dimension length"):
        conv.infer_shapes({"in": (2, 3, 4)}, {"kernel_size": 5, "stride": 1, "padding": 0})

def test_conv1d_boundary_spatial_dimension():
    conv = Conv1DBlock()
    # L=5, kernel=5 -> out_L = (5 - 4 - 1)/1 + 1 = 1
    res = conv.infer_shapes({"in": (2, 3, 5)}, {"in_channels": 3, "out_channels": 8, "kernel_size": 5, "stride": 1, "padding": 0})
    assert res["out"] == (2, 8, 1)

def test_conv1d_lazy_and_auto_infer():
    conv = Conv1DBlock()
    # Auto-infer in_channels from input shape
    res = conv.infer_shapes({"in": (4, 16, 64)}, {"in_channels": -1, "out_channels": 32})
    assert res["out"] == (4, 32, 62)  # default kernel=3, stride=1, pad=0 -> 64 - 2 = 62

    # LAZY in_channels
    res_lazy = conv.infer_shapes({"in": (4, 16, 64)}, {"in_channels": "LAZY", "out_channels": 32})
    assert res_lazy["out"] == (4, 32, 62)
    init_lazy = conv.emit_init("node_1", {"in_channels": "LAZY", "out_channels": 32, "kernel_size": 5})
    assert "nn.LazyConv1d(32, 5" in init_lazy

def test_conv1d_parse_int_1d_robustness():
    assert parse_int_1d(3, 1) == 3
    assert parse_int_1d("5", 1) == 5
    assert parse_int_1d([7], 1) == 7
    assert parse_int_1d((9,), 1) == 9
    assert parse_int_1d("11", 1) == 11
    assert parse_int_1d(None, 2) == 2
    assert parse_int_1d("", 4) == 4
    assert parse_int_1d("None", 6) == 6


# =============================================================================
# 4. EmbeddingBlock Adversarial Tests
# =============================================================================

def test_embedding_rejects_invalid_params():
    emb = EmbeddingBlock()
    with pytest.raises(ValueError, match="num_embeddings must be greater than 0"):
        emb.infer_shapes({"in": (2, 10)}, {"num_embeddings": 0, "embedding_dim": 64})
    with pytest.raises(ValueError, match="num_embeddings must be greater than 0"):
        emb.infer_shapes({"in": (2, 10)}, {"num_embeddings": -5, "embedding_dim": 64})
    with pytest.raises(ValueError, match="embedding_dim must be greater than 0"):
        emb.infer_shapes({"in": (2, 10)}, {"num_embeddings": 100, "embedding_dim": 0})
    with pytest.raises(ValueError, match="embedding_dim must be greater than 0"):
        emb.infer_shapes({"in": (2, 10)}, {"num_embeddings": 100, "embedding_dim": -10})

def test_embedding_shape_inference_dimensions():
    emb = EmbeddingBlock()
    # 1D indices -> (B, embedding_dim)
    assert emb.infer_shapes({"in": (32,)}, {"embedding_dim": 64})["out"] == (32, 64)
    # 2D indices -> (B, L, embedding_dim)
    assert emb.infer_shapes({"in": (4, 32)}, {"embedding_dim": 128})["out"] == (4, 32, 128)
    # 3D indices -> (B, S, L, embedding_dim)
    assert emb.infer_shapes({"in": (2, 4, 32)}, {"embedding_dim": 256})["out"] == (2, 4, 32, 256)
    # ANY shape
    assert emb.infer_shapes({"in": ("ANY",)}, {})["out"] == ("ANY",)
    assert emb.infer_shapes({}, {})["out"] == ("ANY",)

def test_embedding_optional_param_emission():
    emb = EmbeddingBlock()
    code = emb.emit_init("1", {
        "num_embeddings": 5000,
        "embedding_dim": 256,
        "padding_idx": 0,
        "max_norm": 2.5,
        "norm_type": 1.0,
        "scale_grad_by_freq": True,
        "sparse": True,
    })
    assert "nn.Embedding(5000, 256" in code
    assert "padding_idx=0" in code
    assert "max_norm=2.5" in code
    assert "norm_type=1.0" in code
    assert "scale_grad_by_freq=True" in code
    assert "sparse=True" in code


# =============================================================================
# 5. Scalar Binary Operations Adversarial Tests
# =============================================================================

def test_is_valid_scalar_adversarial():
    # Valid scalars
    assert _is_valid_scalar(0) is True
    assert _is_valid_scalar("0") is True
    assert _is_valid_scalar(1) is True
    assert _is_valid_scalar("2.5") is True
    assert _is_valid_scalar("-10") is True
    assert _is_valid_scalar("math.sqrt(self.d_model)") is True
    assert _is_valid_scalar("@var:d_model") is True

    # Invalid scalars
    assert _is_valid_scalar(None) is False
    assert _is_valid_scalar("") is False
    assert _is_valid_scalar("   ") is False
    assert _is_valid_scalar("None") is False
    assert _is_valid_scalar("none") is False
    assert _is_valid_scalar(" NONE ") is False

def test_add_block_scalar_combinations():
    add = AddBlock()
    # Left scalar: 1 + x
    out = add.emit_forward("1", {"in": "x"}, {"out": "res"}, {"scalar_a": "1"})
    assert out == "res = 1 + x"

    # Right scalar: x + 1
    out = add.emit_forward("2", {"in": "x"}, {"out": "res"}, {"scalar_b": "1"})
    assert out == "res = x + 1"

    # Both scalars: 1 + x + 2
    out = add.emit_forward("3", {"in": "x"}, {"out": "res"}, {"scalar_a": "1", "scalar_b": "2"})
    assert out == "res = 1 + x + 2"

    # Multiple tensor inputs + scalars
    out = add.emit_forward("4", {"in": ["x1", "x2"]}, {"out": "res"}, {"scalar_a": "10", "scalar_b": "20"})
    assert out == "res = 10 + x1 + x2 + 20"

    # Ignore invalid scalars
    out = add.emit_forward("5", {"in": "x"}, {"out": "res"}, {"scalar_a": "", "scalar_b": "None"})
    assert out == "res = x"

def test_sub_block_scalar_combinations():
    sub = SubBlock()
    # Left scalar: 10 - x
    out = sub.emit_forward("1", {"in_b": "x"}, {"out": "res"}, {"scalar_a": "10"})
    assert out == "res = 10 - x"

    # Right scalar: x - 5
    out = sub.emit_forward("2", {"in_a": "x"}, {"out": "res"}, {"scalar_b": "5"})
    assert out == "res = x - 5"

    # Both scalars: 10 - 2
    out = sub.emit_forward("3", {}, {"out": "res"}, {"scalar_a": "10", "scalar_b": "2"})
    assert out == "res = 10 - 2"

def test_mul_block_scalar_combinations():
    mul = MulBlock()
    # Left scalar: 2.5 * x
    out = mul.emit_forward("1", {"in": "x"}, {"out": "res"}, {"scalar_a": "2.5"})
    assert out == "res = 2.5 * x"

    # Right scalar: x * 0.125
    out = mul.emit_forward("2", {"in": "x"}, {"out": "res"}, {"scalar_b": "0.125"})
    assert out == "res = x * 0.125"

    # Both scalars: 2 * x * 3
    out = mul.emit_forward("3", {"in": "x"}, {"out": "res"}, {"scalar_a": "2", "scalar_b": "3"})
    assert out == "res = 2 * x * 3"

def test_div_block_scalar_combinations():
    div = DivBlock()
    # Right scalar expression: x / math.sqrt(self.d_model)
    out = div.emit_forward("1", {"in_a": "x"}, {"out": "res"}, {"scalar_b": "math.sqrt(self.d_model)"})
    assert out == "res = x / math.sqrt(self.d_model)"

    # Left scalar expression: 1.0 / x
    out = div.emit_forward("2", {"in_b": "x"}, {"out": "res"}, {"scalar_a": "1.0"})
    assert out == "res = 1.0 / x"

    # Both scalars: 100 / 4
    out = div.emit_forward("3", {}, {"out": "res"}, {"scalar_a": "100", "scalar_b": "4"})
    assert out == "res = 100 / 4"


# =============================================================================
# 6. End-to-End AST Compilation & Execution with Numerical Validation
# =============================================================================

def test_full_pipeline_compilation_and_numerical_execution():
    """
    Constructs a graph containing:
    Input -> Embedding -> Transpose -> Conv1D -> GELU -> SiLU -> Add(scalar) -> Mul(scalar) -> Div(scalar) -> Output
    Compiles to PyTorch code, instantiates module, and executes forward pass.
    Validates torch.allclose against raw torch calculation.
    """
    nodes = [
        Node(id="n_in", data=NodeData(block_id="input", paramValues={"shape": "(2, 16)"})),
        Node(id="n_emb", data=NodeData(block_id="embedding", paramValues={"num_embeddings": 100, "embedding_dim": 32})),
        Node(id="n_trans", data=NodeData(block_id="transpose", paramValues={"dim0": 1, "dim1": 2})),
        Node(id="n_conv", data=NodeData(block_id="conv1d", paramValues={"in_channels": 32, "out_channels": 16, "kernel_size": 3, "padding": 1})),
        Node(id="n_gelu", data=NodeData(block_id="gelu", paramValues={"approximate": "tanh"})),
        Node(id="n_silu", data=NodeData(block_id="silu", paramValues={"inplace": False})),
        Node(id="n_add", data=NodeData(block_id="add", paramValues={"scalar_b": "1.5"})),
        Node(id="n_mul", data=NodeData(block_id="mul", paramValues={"scalar_b": "2.0"})),
        Node(id="n_div", data=NodeData(block_id="div", paramValues={"scalar_b": "math.sqrt(self.d_model)"})),
        Node(id="n_out", data=NodeData(block_id="output")),
    ]

    edges = [
        Edge(id="e1", source="n_in", sourceHandle="out", target="n_emb", targetHandle="in"),
        Edge(id="e2", source="n_emb", sourceHandle="out", target="n_trans", targetHandle="in"),
        Edge(id="e3", source="n_trans", sourceHandle="out", target="n_conv", targetHandle="in"),
        Edge(id="e4", source="n_conv", sourceHandle="out", target="n_gelu", targetHandle="in"),
        Edge(id="e5", source="n_gelu", sourceHandle="out", target="n_silu", targetHandle="in"),
        Edge(id="e6", source="n_silu", sourceHandle="out", target="n_add", targetHandle="in"),
        Edge(id="e7", source="n_add", sourceHandle="out", target="n_mul", targetHandle="in"),
        Edge(id="e8", source="n_mul", sourceHandle="out", target="n_div", targetHandle="in_a"),
        Edge(id="e9", source="n_div", sourceHandle="out", target="n_out", targetHandle="in"),
    ]

    vars_list = [
        ArchVariableModel(id="v_dmodel", name="d_model", type="int", default=16, scope="init_param")
    ]

    graph = GraphData(name="ComplexPipeline", nodes=nodes, edges=edges, variables=vars_list)
    code_res = generate_pytorch_code({"main": graph}, "main")
    code = code_res[0]["main"] if isinstance(code_res, tuple) else code_res["main"]

    # Verify code contents
    assert "import math" in code
    assert "nn.Embedding(100, 32)" in code
    assert "nn.Conv1d(32, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)" in code
    assert "nn.GELU(approximate='tanh')" in code
    assert "nn.SiLU(inplace=False)" in code
    assert "+ 1.5" in code
    assert "* 2.0" in code
    assert "/ math.sqrt(self.d_model)" in code

    # Execute generated code
    namespace = {}
    exec(code, namespace)
    model_cls = namespace.get("ComplexPipeline") or namespace.get("Model")
    assert model_cls is not None

    model = model_cls(d_model=16)
    model.eval()

    # Create fixed input tensor
    indices = torch.randint(0, 100, (2, 16), dtype=torch.long)
    with torch.no_grad():
        out = model(indices)

    assert out.shape == (2, 16, 16)

    # Compute manual oracle step-by-step
    with torch.no_grad():
        x = model.layer_n_emb(indices)
        x = torch.transpose(x, 1, 2)
        x = model.layer_n_conv(x)
        x = model.layer_n_gelu(x)
        x = model.layer_n_silu(x)
        x = x + 1.5
        x = x * 2.0
        x = x / math.sqrt(16)

    assert torch.allclose(out, x, atol=1e-5), f"Numerical mismatch: max diff = {(out - x).abs().max()}"
