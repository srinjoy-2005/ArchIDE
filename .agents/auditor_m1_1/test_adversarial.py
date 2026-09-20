import torch
import torch.nn as nn
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')))
from blocks.activations import GELUBlock, SiLUBlock
from blocks.core import Conv1DBlock, EmbeddingBlock
from blocks.tensor_ops import AddBlock, SubBlock, MulBlock, DivBlock
from blocks import get_block_by_id, get_all_blocks

print("--- Test 1: GELU Block Stress Test ---")
gelu = GELUBlock()
for approx in ["none", "tanh", "TANH", "'tanh'"]:
    init_code = gelu.emit_init("g1", {"approximate": approx})
    print(f"GELU ({approx}) init: {init_code}")
    ns = {}
    exec("import torch.nn as nn\n" + init_code.replace("self.", ""), ns)

for shape in [(2, 10), (1, 3, 224), (2, 4, 8, 16), ("ANY",)]:
    out_s = gelu.infer_shapes({"in": shape}, {})
    assert out_s["out"] == shape, f"Shape mismatch: {out_s}"

print("--- Test 2: SiLU Block Stress Test ---")
silu = SiLUBlock()
for inplace in [True, False]:
    init_code = silu.emit_init("s1", {"inplace": inplace})
    print(f"SiLU ({inplace}) init: {init_code}")
    ns = {}
    exec("import torch.nn as nn\n" + init_code.replace("self.", ""), ns)

print("--- Test 3: Conv1D Block Stress Test ---")
c1d = Conv1DBlock()
# Check mathematical formula for L_out:
# L=101, k=5, s=3, p=2, d=2:
# (101 + 2*2 - 2*(5-1) - 1)//3 + 1 = (101 + 4 - 8 - 1)//3 + 1 = 96//3 + 1 = 33
res = c1d.infer_shapes({"in": (4, 16, 101)}, {"in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 3, "padding": 2, "dilation": 2})
assert res["out"] == (4, 32, 33), f"Expected (4, 32, 33) but got {res}"

# Verify actual PyTorch Conv1d matches shape:
conv_pt = nn.Conv1d(16, 32, 5, stride=3, padding=2, dilation=2)
x = torch.randn(4, 16, 101)
out_pt = conv_pt(x)
assert out_pt.shape == (4, 32, 33), f"PyTorch shape mismatch: {out_pt.shape}"
print("Conv1D math matches PyTorch Conv1d exactly!")

# Check error cases:
try:
    c1d.infer_shapes({"in": (4, 16, 10)}, {"kernel_size": 20, "stride": 1, "padding": 0})
    assert False, "Should have raised negative length error"
except ValueError as e:
    print("Caught expected Conv1D negative length error:", e)

try:
    c1d.infer_shapes({"in": (4, 15, 100)}, {"in_channels": 15, "groups": 2})
    assert False, "Should have raised divisibility error"
except ValueError as e:
    print("Caught expected Conv1D groups divisibility error:", e)

print("--- Test 4: Embedding Block Stress Test ---")
emb = EmbeddingBlock()
# 3D input (e.g. B, N, S)
res = emb.infer_shapes({"in": (2, 8, 16)}, {"num_embeddings": 200, "embedding_dim": 64})
assert res["out"] == (2, 8, 16, 64)
# Real PyTorch test
emb_pt = nn.Embedding(200, 64)
idx = torch.randint(0, 200, (2, 8, 16))
assert emb_pt(idx).shape == (2, 8, 16, 64)
print("Embedding 3D input shape matches PyTorch Embedding exactly!")

print("--- Test 5: Scalar Operations Robustness ---")
add_b = AddBlock()
sub_b = SubBlock()
mul_b = MulBlock()
div_b = DivBlock()

# Edge case: empty string or 'None' should NOT be treated as valid scalar
assert add_b.emit_forward("n1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": ""}) == "y = x"
assert add_b.emit_forward("n1", {"in": ["x"]}, {"out": "y"}, {"scalar_b": "None"}) == "y = x"
assert sub_b.emit_forward("n1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": ""}) == "y = x - None"
assert sub_b.emit_forward("n1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "5.0"}) == "y = x - 5.0"
assert div_b.emit_forward("n1", {"in_a": "x"}, {"out": "y"}, {"scalar_b": "math.sqrt(d)"}) == "y = x / math.sqrt(d)"
assert div_b.emit_forward("n1", {"in_b": "x"}, {"out": "y"}, {"scalar_a": "1.0"}) == "y = 1.0 / x"

print("--- ALL ADVERSARIAL INTEGRITY TESTS PASSED ---")
