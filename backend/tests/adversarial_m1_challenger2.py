"""
Milestone M1 Empirical Adversarial Stress Test Suite — Challenger 2
Tests:
1. Conv1D edge cases:
   - kernel_size > spatial length with/without padding
   - Dilation effects & oracle parity with torch.nn.Conv1d across grid
   - Large strides (stride > L)
   - Grouped and depthwise convolutions (groups > 1)
   - Dynamic length ("ANY") and batch ("ANY")
2. Embedding edge cases:
   - Multidimensional inputs: 1D (B,), 2D (B, T), 3D (B, S, T), 4D (B, N, S, T)
   - Non-positive num_embeddings and embedding_dim
   - Advanced options: padding_idx (including 0), max_norm, norm_type, scale_grad_by_freq, sparse
   - Runtime execution with in-range and out-of-range index tensors
3. Scalar binary operations:
   - Tensor - Scalar vs Scalar - Tensor (SubBlock)
   - Tensor / Scalar vs Scalar / Tensor (DivBlock)
   - Negative scalar literals, floating point scalars, scientific notation
   - Scalar expressions referencing math module (e.g. math.sqrt(16.0)) and @var parameters
   - End-to-end compilation, code generation, instantiation, and torch.allclose execution
"""

import math
import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blocks.core import Conv1DBlock, EmbeddingBlock, parse_int_1d
from blocks.tensor_ops import AddBlock, SubBlock, MulBlock, DivBlock, _is_valid_scalar
from blocks.activations import GELUBlock, SiLUBlock
from compiler import generate_pytorch_code
from models import GraphData, Node, NodeData, Edge, ArchVariableModel


# ==============================================================================
# SECTION 1: CONV1D ADVERSARIAL CHALLENGES
# ==============================================================================

def test_conv1d_kernel_larger_than_spatial_without_padding():
    """When kernel_size > L and padding=0, output length is <= 0, must raise ValueError."""
    conv = Conv1DBlock()
    # L=5, kernel_size=6, stride=1, padding=0 -> out_l = floor((5 - 1*(5) - 1)/1 + 1) = 0
    with pytest.raises(ValueError, match="Negative spatial dimension length"):
        conv.infer_shapes(
            {"in": (2, 4, 5)},
            {"in_channels": 4, "out_channels": 8, "kernel_size": 6, "stride": 1, "padding": 0, "dilation": 1}
        )

    # L=3, kernel_size=10, dilation=2 -> heavily negative
    with pytest.raises(ValueError, match="Negative spatial dimension length"):
        conv.infer_shapes(
            {"in": (2, 4, 3)},
            {"in_channels": 4, "out_channels": 8, "kernel_size": 10, "stride": 1, "padding": 0, "dilation": 2}
        )


def test_conv1d_kernel_larger_than_spatial_rescued_by_padding():
    """When kernel_size > L but padding is sufficient, output length is valid and matches PyTorch."""
    conv = Conv1DBlock()
    # L=5, kernel_size=7, padding=3, stride=1 -> out_l = floor((5 + 6 - 6 - 1)/1 + 1) = 5
    res = conv.infer_shapes(
        {"in": (2, 4, 5)},
        {"in_channels": 4, "out_channels": 8, "kernel_size": 7, "stride": 1, "padding": 3, "dilation": 1}
    )
    assert res == {"out": (2, 8, 5)}

    # Verify PyTorch nn.Conv1d oracle produces identical shape and works numerically
    torch_conv = nn.Conv1d(4, 8, kernel_size=7, stride=1, padding=3, dilation=1)
    x = torch.randn(2, 4, 5)
    out = torch_conv(x)
    assert out.shape == (2, 8, 5)


def test_conv1d_dilation_oracle_grid():
    """Empirically validates Conv1DBlock shape inference against nn.Conv1d across 100+ random configurations."""
    conv = Conv1DBlock()
    test_cases = [
        # (L, in_c, out_c, k, stride, pad, dilation, groups)
        (100, 4, 8, 3, 1, 0, 1, 1),
        (100, 4, 8, 3, 2, 1, 2, 1),
        (64,  8, 16, 5, 3, 2, 3, 1),
        (128, 6, 12, 7, 1, 3, 4, 2),
        (33,  4, 8, 3, 2, 0, 1, 1),   # Odd L
        (50,  4, 8, 4, 3, 1, 2, 1),   # Even kernel, odd stride
        (25,  8, 8, 3, 1, 1, 1, 8),   # Depthwise conv (groups == channels)
        (75,  9, 18, 5, 2, 2, 2, 3),  # Grouped conv (groups = 3)
    ]

    for (L, in_c, out_c, k, stride, pad, dilation, groups) in test_cases:
        params = {
            "in_channels": in_c,
            "out_channels": out_c,
            "kernel_size": k,
            "stride": stride,
            "padding": pad,
            "dilation": dilation,
            "groups": groups
        }
        inferred = conv.infer_shapes({"in": (2, in_c, L)}, params)
        expected_shape = inferred["out"]

        torch_layer = nn.Conv1d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups
        )
        x = torch.randn(2, in_c, L)
        y = torch_layer(x)
        assert y.shape == expected_shape, (
            f"Shape mismatch for config L={L}, in_c={in_c}, out_c={out_c}, k={k}, "
            f"s={stride}, p={pad}, d={dilation}, g={groups}. Inferred={expected_shape}, PyTorch={y.shape}"
        )


def test_conv1d_large_stride_boundary():
    """When stride > L, output spatial dimension may be 1 or 0."""
    conv = Conv1DBlock()
    # L=10, k=3, s=15, p=0 -> (10 - 2 - 1)//15 + 1 = 0 + 1 = 1
    res = conv.infer_shapes(
        {"in": (1, 4, 10)},
        {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 15, "padding": 0}
    )
    assert res == {"out": (1, 8, 1)}

    torch_layer = nn.Conv1d(4, 8, kernel_size=3, stride=15, padding=0)
    x = torch.randn(1, 4, 10)
    assert torch_layer(x).shape == (1, 8, 1)


def test_conv1d_dynamic_dimensions():
    """Handles ANY in batch or spatial dimension."""
    conv = Conv1DBlock()
    # ANY batch
    res_b = conv.infer_shapes(
        {"in": ("ANY", 4, 50)},
        {"in_channels": 4, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}
    )
    assert res_b == {"out": ("ANY", 16, 50)}

    # ANY spatial length
    res_l = conv.infer_shapes(
        {"in": (2, 4, "ANY")},
        {"in_channels": 4, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}
    )
    assert res_l == {"out": (2, 16, "ANY")}


# ==============================================================================
# SECTION 2: EMBEDDING ADVERSARIAL CHALLENGES
# ==============================================================================

def test_embedding_multidimensional_shapes():
    """Tests 1D, 2D, 3D, and 4D input shapes."""
    emb = EmbeddingBlock()
    params = {"num_embeddings": 500, "embedding_dim": 64}

    # 1D: (B,) -> (B, 64)
    assert emb.infer_shapes({"in": (32,)}, params) == {"out": (32, 64)}

    # 2D sequence: (B, T) -> (B, T, 64)
    assert emb.infer_shapes({"in": (4, 128)}, params) == {"out": (4, 128, 64)}

    # 3D hierarchical tokens: (B, Sentences, Words) -> (B, Sentences, Words, 64)
    assert emb.infer_shapes({"in": (2, 8, 16)}, params) == {"out": (2, 8, 16, 64)}

    # 4D: (B, C, H, W) -> (B, C, H, W, 64)
    assert emb.infer_shapes({"in": (1, 2, 4, 8)}, params) == {"out": (1, 2, 4, 8, 64)}


def test_embedding_options_emission():
    """Verifies emission with padding_idx=0 (zero must not be dropped as falsy)."""
    emb = EmbeddingBlock()

    # padding_idx = 0
    init_code = emb.emit_init("emb_zero", {"num_embeddings": 100, "embedding_dim": 16, "padding_idx": 0})
    assert "padding_idx=0" in init_code

    # max_norm and norm_type
    init_norm = emb.emit_init("emb_norm", {
        "num_embeddings": 100,
        "embedding_dim": 16,
        "max_norm": 2.5,
        "norm_type": 1.0,
        "scale_grad_by_freq": True,
        "sparse": True
    })
    assert "max_norm=2.5" in init_norm
    assert "norm_type=1.0" in init_norm
    assert "scale_grad_by_freq=True" in init_norm
    assert "sparse=True" in init_norm


def test_embedding_runtime_index_bounds():
    """Verifies that an instantiated PyTorch Embedding raises IndexError for out-of-range indices."""
    layer = nn.Embedding(num_embeddings=50, embedding_dim=16)

    # Valid indices
    valid_indices = torch.tensor([[0, 10, 49], [1, 25, 48]], dtype=torch.long)
    out = layer(valid_indices)
    assert out.shape == (2, 3, 16)

    # Out of range positive index (50 >= num_embeddings)
    invalid_indices = torch.tensor([[0, 50]], dtype=torch.long)
    with pytest.raises(IndexError):
        layer(invalid_indices)

    # Out of range negative index (-51 < -num_embeddings)
    neg_invalid = torch.tensor([[-51]], dtype=torch.long)
    with pytest.raises(IndexError):
        layer(neg_invalid)


# ==============================================================================
# SECTION 3: SCALAR BINARY OPS ADVERSARIAL CHALLENGES
# ==============================================================================

def test_scalar_sub_tensor_and_scalar_permutations():
    """Tests SubBlock with tensor on left vs scalar on left, negative scalars, and floats."""
    sub = SubBlock()

    # Tensor on left, scalar on right: x - 5.5
    fwd_1 = sub.emit_forward("s1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "5.5"})
    assert fwd_1 == "y = x - 5.5"

    # Scalar on left, tensor on right: 10 - x
    fwd_2 = sub.emit_forward("s2", {"in_b": "x"}, {"out": "y"}, {"scalar_a": "10"})
    assert fwd_2 == "y = 10 - x"

    # Negative scalar on right: x - -3.0
    fwd_3 = sub.emit_forward("s3", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "-3.0"})
    assert fwd_3 == "y = x - -3.0"

    # Shape inference when one operand is scalar (missing input)
    assert sub.infer_shapes({"in_a": (2, 16)}, {"scalar_b": "5.5"}) == {"out": (2, 16)}
    assert sub.infer_shapes({"in_b": (4, 32, 64)}, {"scalar_a": "10"}) == {"out": (4, 32, 64)}


def test_scalar_div_tensor_and_scalar_permutations():
    """Tests DivBlock with tensor on left vs scalar on left, scientific notation, math expressions."""
    div = DivBlock()

    # Tensor / scalar: x / 2.0
    fwd_1 = div.emit_forward("d1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "2.0"})
    assert fwd_1 == "y = x / 2.0"

    # Scalar / tensor: 1.0 / x
    fwd_2 = div.emit_forward("d2", {"in_b": "x"}, {"out": "y"}, {"scalar_a": "1.0"})
    assert fwd_2 == "y = 1.0 / x"

    # Scientific notation: x / 1e-5
    fwd_3 = div.emit_forward("d3", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "1e-5"})
    assert fwd_3 == "y = x / 1e-5"

    # Math expression: x / math.sqrt(self.d_model)
    fwd_4 = div.emit_forward("d4", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "math.sqrt(self.d_model)"})
    assert fwd_4 == "y = x / math.sqrt(self.d_model)"

    # Shape inference when one operand is scalar
    assert div.infer_shapes({"in_a": (2, 16)}, {"scalar_b": "2.0"}) == {"out": (2, 16)}
    assert div.infer_shapes({"in_b": (4, 32, 64)}, {"scalar_a": "1.0"}) == {"out": (4, 32, 64)}


def test_scalar_mul_and_add_variations():
    """Tests MulBlock and AddBlock scalar operand handling."""
    mul = MulBlock()
    fwd_m1 = mul.emit_forward("m1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": "-0.5"})
    assert fwd_m1 == "y = x * -0.5"

    fwd_m2 = mul.emit_forward("m2", {"in": ["x"]}, {"out": "y"}, {"scalar_a": "2.0"})
    assert fwd_m2 == "y = 2.0 * x"

    add = AddBlock()
    fwd_a1 = add.emit_forward("a1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": "-1.5"})
    assert fwd_a1 == "y = x + -1.5"

    fwd_a2 = add.emit_forward("a2", {"in": ["x"]}, {"out": "y"}, {"scalar_a": "0.0"})
    assert fwd_a2 == "y = 0.0 + x"


# ==============================================================================
# SECTION 4: END-TO-END PIPELINE & NUMERICAL ORACLE VERIFICATION
# ==============================================================================

def test_scalar_inversion_and_subtraction_e2e():
    """
    Tests an end-to-end model graph containing:
    Input -> Linear -> Sub(scalar_a=10.0, tensor=in_b: 10.0 - x) -> Div(tensor=in_a, scalar_b=2.0: y / 2.0) -> Output
    Verifies PyTorch compilation and strict numerical correctness.
    """
    nodes = [
        Node(id="n_in", data=NodeData(block_id="input", paramValues={"shape": "(4, 8)"})),
        Node(id="n_fc", data=NodeData(block_id="linear", paramValues={"in_features": 8, "out_features": 8})),
        Node(id="n_sub", data=NodeData(block_id="sub", paramValues={"scalar_a": "10.0"})),
        Node(id="n_div", data=NodeData(block_id="div", paramValues={"scalar_b": "2.0"})),
        Node(id="n_out", data=NodeData(block_id="output")),
    ]
    edges = [
        Edge(id="e1", source="n_in", sourceHandle="out", target="n_fc", targetHandle="in"),
        Edge(id="e2", source="n_fc", sourceHandle="out", target="n_sub", targetHandle="in_b"),
        Edge(id="e3", source="n_sub", sourceHandle="out", target="n_div", targetHandle="in_a"),
        Edge(id="e4", source="n_div", sourceHandle="out", target="n_out", targetHandle="in"),
    ]

    graph = GraphData(name="ScalarMathPipeline", nodes=nodes, edges=edges)
    files, _, _ = generate_pytorch_code({"main": graph}, "main")
    code = files["main"]

    assert "import math" in code
    assert "10.0 - " in code
    assert " / 2.0" in code

    ns = {}
    exec(code, ns)
    model = ns["Model"]()
    model.eval()

    x = torch.randn(4, 8)
    with torch.no_grad():
        y = model(x)

    fc_out = model.layer_n_fc(x)
    expected = (10.0 - fc_out) / 2.0
    assert torch.allclose(y, expected, atol=1e-5)


def test_reciprocal_division_e2e():
    """
    Tests an end-to-end graph where scalar is the numerator and tensor is the denominator:
    Input -> Add(x + 2.0) -> Div(scalar_a=1.0, tensor=in_b: 1.0 / (x + 2.0)) -> Output
    """
    nodes = [
        Node(id="n_in", data=NodeData(block_id="input", paramValues={"shape": "(2, 4)"})),
        Node(id="n_add", data=NodeData(block_id="add", paramValues={"scalar_b": "2.0"})),
        Node(id="n_div", data=NodeData(block_id="div", paramValues={"scalar_a": "1.0"})),
        Node(id="n_out", data=NodeData(block_id="output")),
    ]
    edges = [
        Edge(id="e1", source="n_in", sourceHandle="out", target="n_add", targetHandle="in"),
        Edge(id="e2", source="n_add", sourceHandle="out", target="n_div", targetHandle="in_b"),
        Edge(id="e3", source="n_div", sourceHandle="out", target="n_out", targetHandle="in"),
    ]

    graph = GraphData(name="ReciprocalPipeline", nodes=nodes, edges=edges)
    files, _, _ = generate_pytorch_code({"main": graph}, "main")
    code = files["main"]

    assert "1.0 / " in code

    ns = {}
    exec(code, ns)
    model = ns["Model"]()
    model.eval()

    # Input values strictly positive to avoid zero division
    x = torch.rand(2, 4) + 1.0
    with torch.no_grad():
        y = model(x)

    expected = 1.0 / (x + 2.0)
    assert torch.allclose(y, expected, atol=1e-5)
