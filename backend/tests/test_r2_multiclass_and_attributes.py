import ast
import json
import os
import sys
import pytest

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_decompiler import PyTorchASTDecompiler, decompile_python_to_ir
from agent_compiler import AgentGraphCompiler, compile_ir
from compiler import generate_pytorch_code, shape_inference_multi_graph
from models import GraphData, Node, NodeData, Edge


# ─── 1. Multi-Class Single File Decompilation ──────────────────────────────────

def test_multiclass_single_file_basicblock_resnet():
    """
    Verifies that a single Python file containing both BasicBlock and ResNet classes
    is fully decompiled into distinct, interconnected IRs.
    """
    code = """
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_planes: int = 64, planes: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = BasicBlock(64, 64)
        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x_input):
        feat = self.layer1(x_input)
        out = self.fc(feat)
        return out
"""
    decompiler = PyTorchASTDecompiler()
    all_irs = decompiler.decompile_all_classes(code, file_stem="resnet")

    # 1. Both classes must be extracted
    assert "basicblock" in all_irs
    assert "resnet" in all_irs

    # 2. Submodule IR (BasicBlock) verification
    bb_ir = all_irs["basicblock"]
    assert bb_ir["name"] == "basicblock"
    assert "in" in bb_ir["nodes"]
    assert "conv1" in bb_ir["nodes"]
    assert "bn1" in bb_ir["nodes"]
    assert "relu" in bb_ir["nodes"]
    assert "out" in bb_ir["nodes"]
    assert "in.out -> conv1.in" in bb_ir["edges"]
    assert "conv1.out -> bn1.in" in bb_ir["edges"]
    assert "bn1.out -> relu.in" in bb_ir["edges"]
    assert "relu.out -> out.in" in bb_ir["edges"]

    # 3. Root module IR (ResNet) verification
    resnet_ir = all_irs["resnet"]
    assert resnet_ir["name"] == "resnet"
    assert "in" in resnet_ir["nodes"]
    assert "layer1" in resnet_ir["nodes"]
    assert "fc" in resnet_ir["nodes"]
    assert "out" in resnet_ir["nodes"]

    # Submodule linkage and parameter bindings
    layer1_node = resnet_ir["nodes"]["layer1"]
    assert layer1_node["block"] == "custom_module"
    assert layer1_node["custom_module_id"] == "modules/basicblock"
    assert layer1_node["params"]["in_planes"] == 64
    assert layer1_node["params"]["planes"] == 64

    # Dataflow edges
    assert "in.out -> layer1.in" in resnet_ir["edges"]
    assert "layer1.out -> fc.in" in resnet_ir["edges"]
    assert "fc.out -> out.in" in resnet_ir["edges"]

    # 4. Backward compatibility: decompile_source returns root IR and has all_irs populated
    dec2 = PyTorchASTDecompiler()
    root_ir = dec2.decompile_source(code, file_stem="resnet")
    assert root_ir["name"] == "resnet"
    assert "basicblock" in dec2.all_irs
    assert "resnet" in dec2.all_irs

    # 5. Visual compilation verification
    arch_bb = compile_ir(bb_ir, validate=True)
    assert len(arch_bb["nodes"]) == 5
    arch_resnet = compile_ir(resnet_ir, validate=True)
    assert len(arch_resnet["nodes"]) == 4


def test_multiclass_unet_three_level_hierarchy():
    """
    Tests a 3-level class hierarchy: DoubleConv -> DownBlock -> UNet.
    Verifies topological ordering leaf-first and multi-class custom_module references.
    """
    code = """
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.outc = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        logits = self.outc(x2)
        return logits
"""
    decompiler = PyTorchASTDecompiler()
    all_irs = decompiler.decompile_all_classes(code, file_stem="unet")

    assert set(all_irs.keys()) == {"doubleconv", "downblock", "unet"}

    # DownBlock links to DoubleConv
    down_ir = all_irs["downblock"]
    assert down_ir["nodes"]["conv"]["block"] == "custom_module"
    assert down_ir["nodes"]["conv"]["custom_module_id"] == "modules/doubleconv"
    assert "in.out -> pool.in" in down_ir["edges"]
    assert "pool.out -> conv.in" in down_ir["edges"]
    assert "conv.out -> out.in" in down_ir["edges"]

    # UNet links to DoubleConv and DownBlock
    unet_ir = all_irs["unet"]
    assert unet_ir["nodes"]["inc"]["custom_module_id"] == "modules/doubleconv"
    assert unet_ir["nodes"]["down1"]["custom_module_id"] == "modules/downblock"
    assert "in.out -> inc.in" in unet_ir["edges"]
    assert "inc.out -> down1.in" in unet_ir["edges"]
    assert "down1.out -> outc.in" in unet_ir["edges"]
    assert "outc.out -> out.in" in unet_ir["edges"]


# ─── 2. Deep Object Attribute Chains ──────────────────────────────────────────

def test_deep_attribute_chains_preserves_dataflow():
    """
    Tests arbitrary deep attribute chains on self:
    - self.backbone.layer1(x)
    - self.features[0].conv(x)
    - self.transformer.encoder.layers[0].norm(x)
    - self.blocks[1][0].act(x)
    Verifies that valid node IDs are generated and dataflow edges are unbroken.
    """
    code = """
import torch
import torch.nn as nn

class DeepCompositeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dynamic or external submodules not registered explicitly
        pass

    def forward(self, x):
        x = self.backbone.layer1(x)
        x = self.features[0].conv(x)
        x = self.transformer.encoder.layers[0].norm(x)
        x = self.blocks[1][0].act(x)
        return x
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="deep_model")

    # Verify node creation
    assert "in" in ir["nodes"]
    assert "backbone_layer1" in ir["nodes"]
    assert "features_0_conv" in ir["nodes"]
    assert "transformer_encoder_layers_0_norm" in ir["nodes"]
    assert "blocks_1_0_act" in ir["nodes"]
    assert "out" in ir["nodes"]

    # Verify unbroken dataflow edges from input to output
    assert "in.out -> backbone_layer1.in" in ir["edges"]
    assert "backbone_layer1.out -> features_0_conv.in" in ir["edges"]
    assert "features_0_conv.out -> transformer_encoder_layers_0_norm.in" in ir["edges"]
    assert "transformer_encoder_layers_0_norm.out -> blocks_1_0_act.in" in ir["edges"]
    assert "blocks_1_0_act.out -> out.in" in ir["edges"]

    # Visual compilation must succeed
    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 6
    assert len(arch["edges"]) == 5


def test_sequential_child_layers_and_subscript_invocation():
    """
    Verifies Sequential stores child layers in layer_instances and supports
    subscript calls like self.seq[0](x).
    """
    code = """
import torch
import torch.nn as nn

class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.seq[0](x)
        x = self.seq[1](x)
        x = self.seq[2](x)
        return x
"""
    decompiler = PyTorchASTDecompiler()
    ir = decompiler.decompile_source(code, file_stem="seq_model")

    # Check child layer introspection
    seq_info = decompiler.layer_instances.get("seq")
    assert seq_info is not None
    assert "layers" in seq_info
    assert len(seq_info["layers"]) == 3
    assert seq_info["layers"][0]["block"] == "conv2d"
    assert seq_info["layers"][1]["block"] == "relu"
    assert seq_info["layers"][2]["block"] == "conv2d"

    # Check nodes in IR
    assert "seq_0" in ir["nodes"]
    assert "seq_1" in ir["nodes"]
    assert "seq_2" in ir["nodes"]

    # Check dataflow
    assert "in.out -> seq_0.in" in ir["edges"]
    assert "seq_0.out -> seq_1.in" in ir["edges"]
    assert "seq_1.out -> seq_2.in" in ir["edges"]
    assert "seq_2.out -> out.in" in ir["edges"]

    arch = compile_ir(ir, validate=True)
    assert len(arch["nodes"]) == 5
    assert len(arch["edges"]) == 4


# ─── 3. Cross-File & Recursive Import Resolution ──────────────────────────────

def test_cross_file_recursive_import_decompilation(tmp_path):
    """
    Tests recursive decompilation across multiple files:
    models/block.py defines ConvBlock
    models/classifier.py imports ConvBlock with alias and uses it
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    block_file = models_dir / "block.py"
    block_file.write_text("""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
""", encoding="utf-8")

    classifier_file = models_dir / "classifier.py"
    classifier_file.write_text("""
import torch
import torch.nn as nn
from .block import ConvBlock as BlockAlias

class Classifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block = BlockAlias(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block(x)
        out = self.fc(x)
        return out
""", encoding="utf-8")

    decompiler = PyTorchASTDecompiler(workspace_dir=str(tmp_path))
    with open(classifier_file, "r", encoding="utf-8") as f:
        classifier_code = f.read()

    ir = decompiler.decompile_source(classifier_code, file_stem="classifier", source_path=str(classifier_file))

    # Assert imported ConvBlock was recursively parsed into all_irs
    assert "convblock" in decompiler.all_irs
    convblock_ir = decompiler.all_irs["convblock"]
    assert "conv" in convblock_ir["nodes"]
    assert "relu" in convblock_ir["nodes"]

    # Assert classifier links to modules/convblock
    assert ir["nodes"]["block"]["block"] == "custom_module"
    assert ir["nodes"]["block"]["custom_module_id"] == "modules/convblock"
    assert "in.out -> block.in" in ir["edges"]
    assert "block.out -> fc.in" in ir["edges"]
    assert "fc.out -> out.in" in ir["edges"]


def test_circular_import_safety(tmp_path):
    """
    Tests that circular imports between two local files do not cause infinite recursion.
    """
    file_a = tmp_path / "mod_a.py"
    file_b = tmp_path / "mod_b.py"

    file_a.write_text("""
import torch.nn as nn
from mod_b import ModelB

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = ModelB()
    def forward(self, x):
        return self.b(x)
""", encoding="utf-8")

    file_b.write_text("""
import torch.nn as nn
from mod_a import ModelA

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)
""", encoding="utf-8")

    decompiler = PyTorchASTDecompiler(workspace_dir=str(tmp_path))
    with open(file_a, "r", encoding="utf-8") as f:
        code_a = f.read()

    # Must terminate without RecursionError
    ir_a = decompiler.decompile_source(code_a, file_stem="mod_a", source_path=str(file_a))
    assert ir_a["name"] == "mod_a"
    assert "modela" in decompiler.all_irs or "mod_a" in decompiler.all_irs
    assert "modelb" in decompiler.all_irs or "mod_b" in decompiler.all_irs


# ─── 4. Agent Compiler Port Loading from .ir.json ─────────────────────────────

def test_agent_compiler_load_ports_from_ir_json(tmp_path):
    """
    Verifies that AgentGraphCompiler._load_custom_module_ports can load inputs and outputs
    from an .ir.json file where 'nodes' is a dictionary.
    """
    ir_modules_dir = tmp_path / "ir" / "modules"
    ir_modules_dir.mkdir(parents=True, exist_ok=True)

    dual_in_mod_path = ir_modules_dir / "dual_fusion.ir.json"
    dual_in_mod_data = {
        "name": "dual_fusion",
        "variables": [],
        "nodes": {
            "in": {"block": "input", "var_name": "x_feat"},
            "in_2": {"block": "input", "var_name": "x_context"},
            "cat": {"block": "cat", "params": {"dim": -1}},
            "out": {"block": "output", "var_name": "fused_out"},
            "out_2": {"block": "output", "var_name": "aux_out"}
        },
        "edges": [
            "in.out -> cat.in",
            "in_2.out -> cat.in_2",
            "cat.out -> out.in",
            "cat.out -> out_2.in"
        ]
    }
    dual_in_mod_path.write_text(json.dumps(dual_in_mod_data, indent=2), encoding="utf-8")

    parent_ir = {
        "name": "pipeline",
        "variables": [],
        "nodes": {
            "in": {"block": "input"},
            "in_2": {"block": "input"},
            "fusion": {"block": "custom_module", "custom_module_id": "modules/dual_fusion"},
            "out": {"block": "output"}
        },
        "edges": [
            "in.out -> fusion.in",
            "in_2.out -> fusion.in_2",
            "fusion.out -> out.in"
        ]
    }

    compiler = AgentGraphCompiler(parent_ir, workspace_dir=str(tmp_path))
    inputs, outputs = compiler._load_custom_module_ports("modules/dual_fusion")

    # Inputs should be correctly loaded from dictionary nodes
    assert len(inputs) == 2
    assert inputs[0]["id"] == "in"
    assert inputs[1]["id"] == "in_2"

    # Outputs should be correctly loaded from dictionary nodes
    assert len(outputs) == 2
    assert outputs[0]["id"] == "out"
    assert outputs[1]["id"] == "out_2"

    # Compiling parent IR must produce handles matching the loaded ports
    compiled = compiler.compile()
    fusion_node = next(n for n in compiled["nodes"] if n["id"] == "fusion")
    assert len(fusion_node["data"]["inputs"]) == 2
    assert len(fusion_node["data"]["outputs"]) == 2


# ─── 5. Submodule IR File Export ──────────────────────────────────────────────

def test_decompile_python_to_ir_saves_submodules(tmp_path):
    """
    Tests decompile_python_to_ir writing both the root IR and submodule IRs to disk.
    """
    code = """
import torch
import torch.nn as nn

class SubLayer(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.fc = nn.Linear(size, size)
    def forward(self, x):
        return self.fc(x)

class TopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = SubLayer(32)
        self.head = nn.Linear(32, 10)
    def forward(self, x):
        return self.head(self.sub(x))
"""
    output_path = tmp_path / "ir" / "top_model.ir.json"
    root_ir = decompile_python_to_ir(code, output_path=str(output_path), workspace_dir=str(tmp_path))

    # Verify root file
    assert output_path.exists()
    assert root_ir["name"] == "top_model"

    # Verify submodule file
    sub_path = tmp_path / "ir" / "modules" / "sublayer.ir.json"
    assert sub_path.exists()
    sub_data = json.loads(sub_path.read_text(encoding="utf-8"))
    assert sub_data["name"] == "sublayer"
    assert "fc" in sub_data["nodes"]


# ─── 6. End-to-End PyTorch Code Generation ────────────────────────────────────

def test_end_to_end_multiclass_pytorch_code_generation():
    """
    Verifies that decompiled multiclass IRs can be compiled into GraphData
    and generate valid, executable PyTorch code.
    """
    code = """
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int = 128, d_ff: int = 512):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.ffn = FeedForward(d_model=d_model, d_ff=512)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = self.norm(x)
        return self.ffn(res)
"""
    decompiler = PyTorchASTDecompiler()
    all_irs = decompiler.decompile_all_classes(code, file_stem="transformer_block")

    # Compile each IR to visual .arch JSON
    ffn_arch = compile_ir(all_irs["feedforward"], validate=True)
    block_arch = compile_ir(all_irs["transformerblock"], validate=True)

    # Convert to GraphData instances
    def to_graph_data(arch_dict, name):
        nodes = [
            Node(
                id=n["id"],
                data=NodeData(
                    block_id=n["data"].get("block_id", ""),
                    label=n["data"].get("label", ""),
                    is_functional=n["data"].get("is_functional", False),
                    paramValues=dict(n["data"].get("paramValues", {})),
                    varName=n["data"].get("varName", ""),
                    custom_module_id=n["data"].get("custom_module_id", ""),
                ),
                position=n.get("position"),
            )
            for n in arch_dict["nodes"]
        ]
        edges = [
            Edge(
                id=e["id"],
                source=e["source"],
                sourceHandle=e["sourceHandle"],
                target=e["target"],
                targetHandle=e["targetHandle"],
            )
            for e in arch_dict["edges"]
        ]
        return GraphData(name=name, variables=[], nodes=nodes, edges=edges)

    graphs = {
        "modules/feedforward": to_graph_data(ffn_arch, "FeedForward"),
        "transformer_block": to_graph_data(block_arch, "TransformerBlock"),
    }

    # Generate PyTorch code for the multi-graph system
    compiled_files, _, _ = generate_pytorch_code(graphs, "transformer_block")

    # Verify main model code
    main_code = compiled_files["transformer_block"]
    assert "class TransformerBlock(nn.Module):" in main_code
    assert "self.ffn = FeedForward(" in main_code
    # PyTorch code syntax validation
    parsed = ast.parse(main_code)
    assert isinstance(parsed, ast.Module)

    # Verify submodule code
    ffn_code = compiled_files["modules/feedforward"]
    assert "class FeedForward(nn.Module):" in ffn_code
    parsed_ffn = ast.parse(ffn_code)
    assert isinstance(parsed_ffn, ast.Module)
