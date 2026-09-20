import pytest
import os
import sys

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_decompiler import PyTorchASTDecompiler, decompile_python_to_ir
from agent_compiler import compile_ir, decompile_arch_to_ir


def test_decompile_simple_mlp():
    code = """
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 64):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.layer_fc1 = nn.Linear(self.in_features, self.hidden, bias=True)
        self.layer_relu = nn.ReLU(inplace=False)
        self.layer_fc2 = nn.Linear(self.hidden, 10, bias=True)

    def forward(self, x_input):
        h = self.layer_fc1(x_input)
        a = self.layer_relu(h)
        out = self.layer_fc2(a)
        return out
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="simple_mlp")

    assert ir["name"] == "simple_mlp"
    assert len(ir["variables"]) == 2
    assert ir["variables"][0]["name"] == "in_features"
    assert ir["variables"][0]["default"] == 128

    assert "in" in ir["nodes"]
    assert "fc1" in ir["nodes"]
    assert "relu" in ir["nodes"]
    assert "fc2" in ir["nodes"]
    assert "out" in ir["nodes"]

    assert ir["nodes"]["fc1"]["params"]["in_features"] == "@var:in_features"
    assert ir["nodes"]["fc1"]["params"]["out_features"] == "@var:hidden"

    assert len(ir["edges"]) == 4
    assert "in.out -> fc1.in" in ir["edges"]
    assert "fc1.out -> relu.in" in ir["edges"]
    assert "relu.out -> fc2.in" in ir["edges"]
    assert "fc2.out -> out.in" in ir["edges"]

    # Verify that the decompiled IR compiles cleanly into a visual .arch graph
    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 5
    assert len(arch["edges"]) == 4


def test_decompile_convnet_functional():
    code = """
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.layer_relu = nn.ReLU()
        self.layer_fc = nn.Linear(802816, 10)

    def forward(self, x):
        feat = self.layer_conv(x)
        act = self.layer_relu(feat)
        flat = torch.flatten(act, start_dim=1, end_dim=-1)
        out = self.layer_fc(flat)
        return out
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="convnet")

    assert "in" in ir["nodes"]
    assert "conv" in ir["nodes"]
    assert "relu" in ir["nodes"]
    assert "flat" in ir["nodes"] or "flatten" in ir["nodes"]
    assert "fc" in ir["nodes"]
    assert "out" in ir["nodes"]

    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 6
    assert len(arch["edges"]) == 5


def test_decompile_binary_and_tensor_ops():
    code = """
import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(512)
        self.layer_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        normed = self.layer_norm(x)
        proj = self.layer_proj(normed)
        res = x + proj
        return res
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="res")

    assert "norm" in ir["nodes"]
    assert "proj" in ir["nodes"]
    assert "res" in ir["nodes"]
    assert ir["nodes"]["res"]["block"] == "add"

    # Edges should include x -> res and proj -> res
    assert "in.out -> res.in" in ir["edges"]
    assert "proj.out -> res.in" in ir["edges"]


def test_decompile_submodule_references():
    code = """
import torch
import torch.nn as nn
from modules.attention import Attention
from modules.mlp import Mlp

class Block(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.custom_att = Attention(sample_int=64)
        self.custom_mlp = Mlp(d_model=512, d_ff=2048)

    def forward(self, x):
        a = self.custom_att(x, x, x)
        m = self.custom_mlp(a)
        return m
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="block")

    assert ir["nodes"]["att"]["block"] == "custom_module"
    assert ir["nodes"]["att"]["custom_module_id"] == "modules/attention"
    assert ir["nodes"]["att"]["params"]["sample_int"] == 64

    assert ir["nodes"]["mlp"]["block"] == "custom_module"
    assert ir["nodes"]["mlp"]["custom_module_id"] == "modules/mlp"
    assert ir["nodes"]["mlp"]["params"]["d_model"] == 512


def test_decompile_transformer_block():
    code = """
import torch
import torch.nn as nn
from modules.attention import Attention
from modules.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_k: int = 64, d_v: int = 64, n_heads: int = 8, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.linear_q = nn.Linear(self.d_model, self.d_k)
        self.linear_k = nn.Linear(self.d_model, self.d_k)
        self.linear_v = nn.Linear(self.d_model, self.d_v)
        self.attention = Attention()
        self.out_proj = nn.Linear(self.d_k, self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)
        self.mlp = MLP(d_model=self.d_model, d_ff=self.d_ff)

    def forward(self, x_input):
        norm_1 = self.norm_1(x_input)
        q = self.linear_q(norm_1)
        k = self.linear_k(norm_1)
        v = self.linear_v(norm_1)
        attn = self.attention(q, k, v)
        out_proj = self.out_proj(attn)
        res_1 = x_input + out_proj
        norm_2 = self.norm_2(res_1)
        mlp_out = self.mlp(norm_2)
        res_2 = res_1 + mlp_out
        return res_2
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="transformer_block")

    assert len(ir["nodes"]) == 12
    assert "in" in ir["nodes"]
    assert "norm_1" in ir["nodes"]
    assert "linear_q" in ir["nodes"]
    assert "linear_k" in ir["nodes"]
    assert "linear_v" in ir["nodes"]
    assert "attention" in ir["nodes"]
    assert "out_proj" in ir["nodes"]
    assert "res_1" in ir["nodes"]
    assert "norm_2" in ir["nodes"]
    assert "mlp" in ir["nodes"]
    assert "res_2" in ir["nodes"]
    assert "out" in ir["nodes"]

    # Verify multi-input attention handles
    assert "linear_q.out -> attention.in" in ir["edges"]
    assert "linear_k.out -> attention.in_2" in ir["edges"]
    assert "linear_v.out -> attention.in_3" in ir["edges"]

    # Verify residual connections
    assert "in.out -> res_1.in" in ir["edges"]
    assert "out_proj.out -> res_1.in" in ir["edges"]
    assert "res_1.out -> res_2.in" in ir["edges"]
    assert "mlp.out -> res_2.in" in ir["edges"]

    # Verify it compiles cleanly to .arch
    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 12
    assert len(arch["edges"]) == 15


def test_decompile_file4_multihead_fusion():
    code = """
import torch
import torch.nn as nn

class File4(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.attention_pool = MultiheadAttentionPooling(embed_dim=self.embed_dim, num_heads=8)
        self.fusion = GatedFusion(embed_dim=self.embed_dim)
        self.refinement = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x_input):
        attention_features = self.attention_pool(x_input)
        x = self.fusion(attention_features, x_input)
        x = self.refinement(x) + x
        out = self.classifier(x)
        return out
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="file4")

    assert len(ir["nodes"]) == 7
    assert "in" in ir["nodes"]
    assert "attention_pool" in ir["nodes"]
    assert "fusion" in ir["nodes"]
    assert "refinement" in ir["nodes"]
    assert "classifier" in ir["nodes"]
    assert "out" in ir["nodes"]

    # Verify fusion handles both attention features and input
    assert "attention_pool.out -> fusion.in" in ir["edges"]
    assert "in.out -> fusion.in_2" in ir["edges"]

    # Verify compile to arch
    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 7

